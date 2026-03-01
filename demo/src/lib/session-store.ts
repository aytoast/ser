/**
 * Client-side session cache for uploaded audio + diarization results.
 * Uses module-level Map for File/Blob URL (can't be serialized) and
 * sessionStorage for JSON data (survives Next.js client-side navigation).
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
  face_emotion?: string  // FER result (video only): Anger | Contempt | Disgust | Fear | Happy | Neutral | Sad | Surprise
}

export type DiarizeResult = {
  segments: Segment[]
  duration: number
  text: string
  filename: string
  diarization_method?: string
  has_video?: boolean    // true when FER was performed on video frames
}

type SessionEntry = {
  data: DiarizeResult
  audioUrl: string   // blob: URL — valid for this browser session
  filename: string
}

// In-memory store (lives as long as the page/tab is open)
const _store = new Map<string, SessionEntry>()

function _sessionKey(id: string) {
  return `ser-session:${id}`
}

/** Store a new session; returns the session ID. */
export function createSession(file: File, data: DiarizeResult): string {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const audioUrl = URL.createObjectURL(file)
  _store.set(id, { data, audioUrl, filename: file.name })

  // Persist JSON to sessionStorage for resilience across navigation
  try {
    sessionStorage.setItem(_sessionKey(id), JSON.stringify({ data, filename: file.name }))
  } catch {
    // sessionStorage quota exceeded — in-memory only
  }

  return id
}

/** Retrieve a session. Returns null if not found. */
export function getSession(id: string): SessionEntry | null {
  if (!id) return null

  // 1. Try in-memory (audio URL still valid)
  const entry = _store.get(id)
  if (entry) return entry

  // 2. Fall back to sessionStorage (no audio URL after page reload)
  try {
    const raw = sessionStorage.getItem(_sessionKey(id))
    if (raw) {
      const { data, filename } = JSON.parse(raw) as { data: DiarizeResult; filename: string }
      return { data, audioUrl: "", filename }
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
