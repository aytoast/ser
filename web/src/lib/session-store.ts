/**
 * Client-side session cache for uploaded audio + diarization results.
 * Uses module-level Map for File/Blob URL (can't be serialized) and
 * sessionStorage for JSON data (survives Next.js client-side navigation).
 * Also maintains a localStorage-backed recent sessions list.
 *
 * Only import in "use client" components — this module uses browser APIs.
 */

export type Segment = {
  id: number
  speaker: string    // e.g. "SPEAKER_00"
  start: number      // seconds
  end: number        // seconds
  text: string
  emotion: string
  valence: number
  arousal: number
  face_emotion?: string  // MobileViT-XXS FER — present only for video inputs
}

export type DiarizeResult = {
  segments: Segment[]
  duration: number
  text: string
  filename: string
  diarization_method?: string
  has_video?: boolean    // true when FER ran on a video input
}

type SessionEntry = {
  id: string
  data: DiarizeResult
  audioUrl: string   // blob: URL — valid for this browser session
  filename: string
  file?: File        // Original file (only in-memory)
}

// In-memory store (lives as long as the page/tab is open)
const _store = new Map<string, SessionEntry>()

function _sessionKey(id: string) {
  return `ser-session:${id}`
}

// ─── Recent Sessions Registry ─────────────────────────────────────────────────

export type RecentSession = {
  id: string
  filename: string
  duration: number       // seconds
  speakerCount: number
  createdAt: number      // Date.now()
}

const RECENT_KEY = "ser-recent-sessions"
const MAX_RECENT = 20

export function listRecentSessions(): RecentSession[] {
  try {
    const raw = localStorage.getItem(RECENT_KEY)
    return raw ? (JSON.parse(raw) as RecentSession[]) : []
  } catch {
    return []
  }
}

function _pushRecentSession(entry: RecentSession) {
  try {
    const existing = listRecentSessions().filter((s) => s.id !== entry.id)
    const updated = [entry, ...existing].slice(0, MAX_RECENT)
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated))
  } catch {
    // ignore quota errors
  }
}

export function removeRecentSession(id: string) {
  try {
    const updated = listRecentSessions().filter((s) => s.id !== id)
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated))
  } catch {
    // ignore
  }
}

// ─── Initialization & Demo Data ─────────────────────────────────────────────

const INIT_KEY = "ser-app-initialized"

const DEMO_DATA: DiarizeResult = {
  filename: "Welcome Demo.mp4",
  duration: 42.5,
  text: "Welcome to Ethos Studio! This is a demo session to show you how emotional speech recognition works.",
  segments: [
    { id: 1, speaker: "SPEAKER_01", start: 0, end: 5.2, text: "Welcome to the future of emotional intelligence.", emotion: "Trust", valence: 0.8, arousal: 0.3 },
    { id: 2, speaker: "SPEAKER_01", start: 6.5, end: 12.8, text: "With Voxtral, we can now transcribe and analyze the tone of every conversation.", emotion: "Inspiration", valence: 0.9, arousal: 0.6 },
    { id: 3, speaker: "SPEAKER_02", start: 14.1, end: 18.5, text: "That sounds incredibly powerful for sales and companionship applications.", emotion: "Interest", valence: 0.6, arousal: 0.4 },
    { id: 4, speaker: "SPEAKER_01", start: 20.2, end: 28.4, text: "Exactly. It's about understanding the 'how' behind the 'what'.", emotion: "Confidence", valence: 0.7, arousal: 0.2 },
  ]
}

/** 
 * Call this on app start. 
 * 1. Discards localStorage if this is a fresh browser session (mimics server restart cleanup).
 * 2. Prepopulates with demo if empty.
 */
export function ensureStoreInitialized() {
  if (typeof window === "undefined") return

  // 1. Detect fresh session
  const isInitialized = sessionStorage.getItem(INIT_KEY)
  if (!isInitialized) {
    // Clear everything for a clean start
    localStorage.removeItem(RECENT_KEY)

    // Clear all session cache keys from sessionStorage (since localStorage doesn't store them)
    // Actually we use _sessionKey(id) which is stored in sessionStorage
    for (const key in sessionStorage) {
      if (key.startsWith("ser-session:")) sessionStorage.removeItem(key)
    }

    sessionStorage.setItem(INIT_KEY, "true")
  }

  // 2. Prepopulate if empty
  const current = listRecentSessions()
  if (current.length === 0) {
    const demoId = "welcome-demo"

    // Store data in sessionStorage fallback so it works immediately
    try {
      sessionStorage.setItem(_sessionKey(demoId), JSON.stringify({
        data: DEMO_DATA,
        filename: DEMO_DATA.filename
      }))
    } catch { }

    _pushRecentSession({
      id: demoId,
      filename: DEMO_DATA.filename,
      duration: DEMO_DATA.duration,
      speakerCount: 2,
      createdAt: Date.now(),
    })
  }
}

// ─── Session Store API ────────────────────────────────────────────────────────

/** Create an initial session with just the file. Returns ID. */
export function createPendingSession(file: File): string {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const audioUrl = URL.createObjectURL(file)

  const emptyData: DiarizeResult = {
    segments: [],
    duration: 0,
    text: "",
    filename: file.name
  }

  _store.set(id, { id, data: emptyData, audioUrl, filename: file.name, file })

  // Initial persist to sessionStorage
  try {
    sessionStorage.setItem(_sessionKey(id), JSON.stringify({ data: emptyData, filename: file.name }))
  } catch { }

  // Initial register in recents
  _pushRecentSession({
    id,
    filename: file.name,
    duration: 0,
    createdAt: Date.now(),
    speakerCount: 0,
  })

  return id
}

/** Update an existing session with results. */
export function updateSession(id: string, data: DiarizeResult) {
  const entry = _store.get(id)
  if (!entry) return

  const updatedEntry = { ...entry, data }
  _store.set(id, updatedEntry)

  // Persist to sessionStorage
  try {
    sessionStorage.setItem(_sessionKey(id), JSON.stringify({ data, filename: entry.filename }))
  } catch { }

  // Update recents
  const speakers = new Set(data.segments.map((s) => s.speaker))
  _pushRecentSession({
    id,
    filename: entry.filename,
    duration: data.duration,
    createdAt: Date.now(),
    speakerCount: speakers.size,
  })
}

/** Store a new session; returns the session ID. */
export function createSession(file: File, data: DiarizeResult): string {
  const id = createPendingSession(file)
  updateSession(id, data)
  return id
}

/** Retrieve a session. Returns null if not found. */
export function getSession(id: string): SessionEntry | null {
  if (!id) return null

  // Special case: Built-in Welcome Demo
  if (id === "welcome-demo") {
    return {
      id: "welcome-demo",
      data: DEMO_DATA,
      audioUrl: "https://www.w3schools.com/html/horse.mp3", // Built-in sample
      filename: DEMO_DATA.filename,
    }
  }

  // 1. Try in-memory (audio URL still valid)
  const entry = _store.get(id)
  if (entry) return entry

  // 2. Fall back to sessionStorage (no audio URL after page reload)
  try {
    const raw = sessionStorage.getItem(_sessionKey(id))
    if (raw) {
      const { data, filename } = JSON.parse(raw) as { data: DiarizeResult; filename: string }
      return { id, data, audioUrl: "", filename }
    }
  } catch {
    // ignore parse errors
  }

  return null
}

/** Release resources for a session. */
export function clearSession(id: string) {
  const entry = _store.get(id)
  if (entry?.audioUrl) {
    URL.revokeObjectURL(entry.audioUrl)
  }
  _store.delete(id)
  try {
    sessionStorage.removeItem(_sessionKey(id))
  } catch {
    // ignore
  }
}
