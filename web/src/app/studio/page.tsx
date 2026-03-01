"use client"

import React, { useState, useRef, useEffect, useMemo, Suspense } from "react"
import Link from "next/link"
import { useSearchParams, useRouter } from "next/navigation"
import {
  ArrowLeft, ArrowCounterClockwise, ArrowClockwise,
  Export, Play, Pause, Plus, DotsThreeVertical,
  MinusCircle, PlusCircle, Waveform, VideoCamera,
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Progress } from "@/components/ui/progress"
import { getSession, updateSession, type Segment, type DiarizeResult } from "@/lib/session-store"
import { Navbar } from "@/components/navbar"
import { cn } from "@/lib/utils"
import { MagnifyingGlass } from "@phosphor-icons/react"
import NextImage from "next/image"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000"

// --- Constants ---
const SPEAKER_COLORS = [
  { avatar: "bg-blue-400", track: "bg-blue-200" },
  { avatar: "bg-pink-400", track: "bg-pink-200" },
  { avatar: "bg-emerald-400", track: "bg-emerald-200" },
  { avatar: "bg-amber-400", track: "bg-amber-200" },
  { avatar: "bg-violet-400", track: "bg-violet-200" },
  { avatar: "bg-cyan-400", track: "bg-cyan-200" },
  { avatar: "bg-rose-400", track: "bg-rose-200" },
  { avatar: "bg-lime-400", track: "bg-lime-200" },
]

// --- Mock Data (demo / fallback) ---
const MOCK_SEGMENTS: Segment[] = [
  { id: 1, speaker: "Theodore", start: 0.10, end: 7.28, text: "[Instrumental music plays]", emotion: "Neutral", valence: 0.0, arousal: 0.1 },
  { id: 2, speaker: "Theodore", start: 8.10, end: 9.04, text: "Hello, I'm here.", emotion: "Calm", valence: 0.3, arousal: -0.1 },
  { id: 3, speaker: "Samantha", start: 10.62, end: 11.00, text: "Oh.", emotion: "Surprise", valence: 0.4, arousal: 0.6 },
  { id: 4, speaker: "Samantha", start: 13.82, end: 14.18, text: "Hi.", emotion: "Neutral", valence: 0.1, arousal: 0.0 },
  { id: 5, speaker: "Theodore", start: 14.82, end: 16.40, text: "Hi.", emotion: "Happy", valence: 0.7, arousal: 0.4 },
]
const MOCK_FILENAME = "WeChat_20250804025710.mp4"
const MOCK_DURATION = 28.50

// --- Helpers ---
function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(2)
  return m > 0 ? `${m}:${s.padStart(5, "0")}` : s
}

type SpeakerInfo = { label: string; avatarColor: string; trackColor: string }

function buildSpeakerMap(segments: Segment[]): Record<string, SpeakerInfo> {
  const speakers = [...new Set(segments.map(s => s.speaker))].sort()
  return Object.fromEntries(
    speakers.map((id, i) => {
      const palette = SPEAKER_COLORS[i % SPEAKER_COLORS.length]
      // Use the speaker ID as the label if it's a name, otherwise "Speaker N"
      const label = id.startsWith("SPEAKER_") ? `Speaker ${i + 1}` : id
      return [id, { label, avatarColor: palette.avatar, trackColor: palette.track }]
    })
  )
}

// --- SegmentRow ---
function SegmentRow({
  seg,
  active,
  speaker,
  onClick,
}: {
  seg: Segment
  active: boolean
  speaker: SpeakerInfo
  onClick: () => void
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "flex items-start gap-12 group transition-all duration-300 cursor-pointer py-4 rounded-lg px-4 border border-transparent",
        active ? "bg-accent/[0.04] border-accent/10" : "hover:bg-muted/30"
      )}
    >
      {/* Speaker */}
      <div className="w-32 flex items-center gap-3 shrink-0 pt-1">
        <Avatar className="size-6 ring-2 ring-background border border-border/20">
          <AvatarFallback className={cn("text-[10px] text-white", speaker.avatarColor)}>
            {speaker.label[0]}
          </AvatarFallback>
        </Avatar>
        <span className="text-[13px] font-semibold text-foreground/70">{speaker.label}</span>
      </div>

      {/* Content */}
      <div className="flex-1 space-y-1.5 min-w-0 border-l-[1.5px] border-border/30 pl-8 relative">
        <span className="text-[10px] font-sans font-medium text-muted-foreground/40 block tracking-tight">{fmtTime(seg.start)}</span>
        <p className="text-[15px] text-foreground leading-[1.6] font-medium tracking-tight whitespace-pre-wrap">{seg.text}</p>
        <div className="flex items-center gap-2 flex-wrap pt-0.5">
          {seg.emotion && (
            <Badge variant="secondary" className="text-[10px] h-5 px-2 font-medium rounded-full">
              {seg.emotion}
            </Badge>
          )}
          {seg.face_emotion && (
            <Badge variant="outline" className="text-[10px] h-5 px-2 font-medium rounded-full gap-1">
              <VideoCamera size={9} weight="fill" className="text-pink-500" />
              {seg.face_emotion}
            </Badge>
          )}
        </div>
        <span className="text-[10px] font-sans font-medium text-muted-foreground/40 block tracking-tight">{fmtTime(seg.end)}</span>
      </div>
    </div>
  )
}

function MergeButton() {
  return (
    <div className="flex items-center gap-12 px-4 py-0.5 group/merge h-6">
      <div className="w-32 flex justify-end pr-5 shrink-0">
        <Tooltip>
          <TooltipTrigger asChild>
            <button className="text-muted-foreground/20 hover:text-foreground transition-all duration-200 transform hover:scale-125">
              <svg width="14" height="14" viewBox="0 0 256 256" fill="currentColor">
                <path d="M205.66,117.66l-32,32a8,8,0,0,1-11.32-11.32L188.69,112l-26.35-26.34a8,8,0,0,1,11.32-11.32l32,32A8,8,0,0,1,205.66,117.66ZM81.66,138.34,55.31,112,81.66,85.66a8,8,0,0,0-11.32-11.32l-32,32a8,8,0,0,0,0,11.32l32,32a8,8,0,0,0,11.32-11.32Z" />
              </svg>
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" className="bg-black text-white border-black">Merge segments</TooltipContent>
        </Tooltip>
      </div>
      <div className="flex-1 h-px bg-border/20" />
    </div>
  )
}

// --- RightPanel ---
function RightPanel({
  filename,
  onToggle,
}: {
  activeSegment: Segment | null
  audioUrl: string
  filename: string
  isPlaying: boolean
  currentTime: number
  duration: number
  onToggle: () => void
}) {
  return (
    <div className="flex flex-col h-full border-l border-border bg-background">
      {/* Video Preview */}
      <div className="aspect-video w-full bg-slate-950 flex items-center justify-center flex-shrink-0 relative group">
        <NextImage src="/logo.svg" alt="Preview" width={48} height={48} className="opacity-10" />
        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <Button variant="ghost" size="icon" className="text-white hover:bg-white/20" onClick={onToggle}>
            <Play size={24} weight="fill" />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-6 space-y-8">
          {/* Global Properties */}
          <div className="space-y-4">
            <button className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground uppercase tracking-widest hover:text-foreground transition-colors w-full text-left">
              <span className="text-[8px]">▼</span> Global Properties
            </button>
            <div className="space-y-4 pt-2">
              <div className="text-sm font-medium">{filename}</div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Language</span>
                <span className="font-medium text-foreground">English</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Subtitles</span>
                <Button variant="ghost" size="icon" className="h-4 w-4 text-muted-foreground">
                  <DotsThreeVertical size={14} />
                </Button>
              </div>
            </div>
          </div>


        </div>
      </ScrollArea>
    </div>
  )
}

// --- TimelineBar ---
function TimelineBar({
  isPlaying,
  onToggle,
  segments,
  duration,
  currentTime,
  speakerMap,
}: {
  isPlaying: boolean
  onToggle: () => void
  segments: Segment[]
  duration: number
  currentTime: number
  speakerMap: Record<string, SpeakerInfo>
}) {
  const speakers = Object.entries(speakerMap)

  return (
    <div className="border-t border-border bg-background flex-shrink-0">
      <div className="flex overflow-hidden">
        {/* Speakers Sidebar */}
        <div className="w-48 flex-shrink-0 border-r border-border">
          <div className="px-4 h-12 flex items-center justify-between border-b border-border/40">
            <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Speakers</span>
            <Button variant="ghost" size="icon" className="size-5">
              <Plus size={14} />
            </Button>
          </div>
          <div>
            {speakers.map(([id, info]) => (
              <div key={id} className="h-11 px-3 flex items-center gap-2 group hover:bg-muted/30 transition-colors">
                <DotsThreeVertical size={14} className="text-muted-foreground opacity-30 cursor-grab" />
                <Avatar className="size-5">
                  <AvatarFallback className={cn("text-[8px] text-white", info.avatarColor)}>
                    {info.label[0]}
                  </AvatarFallback>
                </Avatar>
                <span className="text-xs font-medium truncate flex-1">{info.label}</span>
                <Button variant="ghost" size="icon" className="size-5 opacity-0 group-hover:opacity-100">
                  <DotsThreeVertical size={14} />
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* Tracks Area */}
        <div className="flex-1 relative overflow-hidden">
          {/* Time Marks */}
          <div className="h-12 border-b border-border/40 flex items-center relative px-4">
            <Badge variant="secondary" className="bg-black text-white text-[10px] h-5 px-1.5 rounded-[4px] absolute left-4 z-10">0.00</Badge>
            {[5, 10, 15, 20].map(s => (
              <div key={s} className="absolute text-[10px] text-muted-foreground/40 font-sans border-l border-border/40 h-2 top-1/2 -translate-y-1/2" style={{ left: `${(s / duration) * 100}%` }}>
                <span className="ml-1 relative -top-3">0:{s.toString().padStart(2, '0')}</span>
              </div>
            ))}
          </div>

          {/* Track Grid */}
          <div className="relative">
            {speakers.map(([id, info]) => (
              <div key={id} className="h-11 border-b border-border/20 relative">
                {duration > 0 && segments.filter(s => s.speaker === id).map(seg => (
                  <div
                    key={seg.id}
                    className={cn("absolute top-2 bottom-2 rounded-[6px] transition-opacity hover:opacity-80 cursor-alias", info.trackColor)}
                    style={{
                      left: `${(seg.start / duration) * 100}%`,
                      width: `${Math.max(((seg.end - seg.start) / duration) * 100, 0.5)}%`,
                    }}
                  />
                ))}
              </div>
            ))}

            {/* Playhead */}
            {duration > 0 && (
              <div
                className="absolute top-0 bottom-0 w-px bg-black z-20 pointer-events-none transition-all duration-75"
                style={{
                  left: (currentTime / duration) * 100 + "%"
                }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="flex items-center justify-between px-6 h-12 border-t border-border/40">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <MagnifyingGlass size={14} className="text-muted-foreground" />
            <Slider defaultValue={[40]} max={100} className="w-32" />
            <MagnifyingGlass size={14} className="text-muted-foreground" />
          </div>
        </div>

        <div className="flex items-center gap-6">
          <span className="text-[13px] font-bold text-foreground">1.0x</span>
          <button
            onClick={onToggle}
            className="size-8 rounded-full bg-black text-white flex items-center justify-center hover:scale-105 transition-transform"
          >
            {isPlaying ? <Pause size={16} weight="fill" /> : <Play size={16} weight="fill" className="ml-0.5" />}
          </button>
        </div>

        <Button variant="ghost" size="sm" className="text-xs font-bold hover:bg-transparent">
          Add segment
        </Button>
      </div>
    </div>
  )
}

// --- Studio Content ---
function StudioContent() {
  const searchParams = useSearchParams()
  const sessionId = searchParams.get("s")
  const router = useRouter()

  const audioRef = useRef<HTMLAudioElement>(null)

  const [session, setSession] = useState(() => sessionId ? getSession(sessionId) : null)
  const [activeId, setActiveId] = useState<number>(1)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processError, setProcessError] = useState<string | null>(null)

  // Sync session state with sessionId param
  useEffect(() => {
    if (sessionId) {
      const s = getSession(sessionId)
      setSession(s)
      if (s?.data.segments && s.data.segments.length > 0) {
        setActiveId(s.data.segments[0].id)
      }
    }
  }, [sessionId])

  // Automatic processing for pending sessions
  useEffect(() => {
    if (!session || isProcessing || processError) return

    // If we have a file but no segments, it's a pending session
    if (session.file && session.data.segments.length === 0) {
      const process = async () => {
        setIsProcessing(true)
        setProcessError(null)
        try {
          const formData = new FormData()
          formData.append("audio", session.file!, session.filename)

          const res = await fetch(`${API_BASE}/api/transcribe-diarize`, {
            method: "POST",
            body: formData,
          })

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}))
            throw new Error(errData.error ?? "Processing failed")
          }

          const data = await res.json() as DiarizeResult
          updateSession(session.id, data)

          // Re-fetch to update local state and trigger re-render
          const updated = getSession(session.id)
          setSession(updated)
          if (updated?.data.segments && updated.data.segments.length > 0) {
            setActiveId(updated.data.segments[0].id)
          }
        } catch (e) {
          setProcessError(e instanceof Error ? e.message : "Request failed")
        } finally {
          setIsProcessing(false)
        }
      }
      process()
    }
  }, [session, isProcessing, processError])

  const segments = session?.data.segments ?? (session ? [] : MOCK_SEGMENTS)
  const audioUrl = session?.audioUrl ?? ""
  const filename = session?.filename ?? MOCK_FILENAME
  const duration = session?.data.duration ?? MOCK_DURATION

  const speakerMap = useMemo(() => buildSpeakerMap(segments), [segments])
  const activeSegment = segments.find(s => s.id === activeId) ?? segments[0] ?? null

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onTime = () => setCurrentTime(audio.currentTime)
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    const onLoadedMetadata = () => {
      // If session duration is 0 (pending), update it from audio metadata
      if (duration === 0) {
        // We could update the store here, but let's just use it locally for now
      }
    }
    audio.addEventListener("timeupdate", onTime)
    audio.addEventListener("play", onPlay)
    audio.addEventListener("pause", onPause)
    audio.addEventListener("loadedmetadata", onLoadedMetadata)
    return () => {
      audio.removeEventListener("timeupdate", onTime)
      audio.removeEventListener("play", onPlay)
      audio.removeEventListener("pause", onPause)
      audio.removeEventListener("loadedmetadata", onLoadedMetadata)
    }
  }, [audioUrl, duration])

  const handleSegmentClick = (seg: Segment) => {
    setActiveId(seg.id)
    if (audioRef.current && audioUrl) {
      audioRef.current.currentTime = seg.start
    }
  }

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio || !audioUrl) return
    if (isPlaying) audio.pause()
    else audio.play()
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Hidden audio element */}
      <audio ref={audioRef} src={audioUrl || undefined} preload="metadata" className="hidden" />

      {/* Top Bar */}
      <Navbar
        breadcrumbs={[
          { label: filename }
        ]}
        actions={
          <div className="flex items-center gap-2">
            {isProcessing && (
              <Badge variant="secondary" className="bg-blue-500/10 text-blue-500 hover:bg-blue-500/10 border-blue-500/20 gap-2 font-medium px-3 h-8">
                <div className="size-2 rounded-full bg-blue-500 animate-pulse" />
                Analysing Speech...
              </Badge>
            )}

            <div className="flex items-center gap-0.5 border border-border rounded-[12px] p-0.5 bg-muted/20 ml-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-foreground hover:bg-background rounded-[8px]">
                    <ArrowCounterClockwise size={16} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Undo</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-foreground hover:bg-background rounded-[8px]">
                    <ArrowClockwise size={16} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Redo</TooltipContent>
              </Tooltip>
            </div>

            <Separator orientation="vertical" className="h-4 mx-1" />

            <Button className="gap-2 font-semibold h-8 text-xs px-4 rounded-[12px] bg-foreground text-background hover:bg-foreground/90 shadow-none border border-transparent">
              <Export size={16} weight="bold" />
              Export
            </Button>
          </div>
        }
      />

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Transcript */}
        <div className="flex-1 overflow-hidden flex flex-col min-w-0 relative">
          {isProcessing && segments.length === 0 && (
            <div className="absolute inset-0 z-20 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center animate-in fade-in duration-500">
              <div className="size-16 mb-6 relative">
                <div className="absolute inset-0 border-4 border-muted rounded-full" />
                <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin" />
                <Waveform size={24} className="absolute inset-0 m-auto text-primary animate-pulse" />
              </div>
              <h2 className="text-xl font-bold mb-2">Analysing Audio</h2>
              <p className="text-muted-foreground text-sm max-w-xs mx-auto">
                Voxtral is currently transcribing and identifying speakers. This should only take a moment...
              </p>
            </div>
          )}

          {processError && (
            <div className="absolute inset-0 z-20 bg-background flex flex-col items-center justify-center p-8 text-center">
              <div className="size-12 rounded-full bg-destructive/10 text-destructive flex items-center justify-center mb-4">
                <Export size={24} className="rotate-45" /> {/* Use as X icon */}
              </div>
              <h2 className="text-xl font-bold mb-2">Processing Failed</h2>
              <p className="text-muted-foreground text-sm max-w-sm mx-auto mb-6">
                {processError}
              </p>
              <Button onClick={() => window.location.reload()} variant="outline">Try Again</Button>
            </div>
          )}

          <ScrollArea className="flex-1">
            <div className="max-w-none px-[200px] py-10">
              {segments.length === 0 && !isProcessing && !processError && (
                <div className="py-20 text-center text-muted-foreground">
                  No segments found.
                </div>
              )}
              {segments.map((seg, i) => (
                <React.Fragment key={seg.id}>
                  {i > 0 && segments[i - 1].speaker === seg.speaker && <MergeButton />}
                  <SegmentRow
                    seg={seg}
                    active={seg.id === activeId}
                    speaker={speakerMap[seg.speaker]}
                    onClick={() => handleSegmentClick(seg)}
                  />
                </React.Fragment>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Right: Properties panel */}
        <div className="w-[320px] flex-shrink-0 overflow-hidden flex flex-col">
          <RightPanel
            activeSegment={activeSegment}
            audioUrl={audioUrl}
            filename={filename}
            isPlaying={isPlaying}
            currentTime={currentTime}
            duration={duration}
            onToggle={togglePlay}
          />
        </div>
      </div>

      {/* Bottom: Timeline */}
      <TimelineBar
        isPlaying={isPlaying}
        onToggle={togglePlay}
        segments={segments}
        duration={duration}
        currentTime={currentTime}
        speakerMap={speakerMap}
      />
    </div>
  )
}

// --- Page (wraps in Suspense for useSearchParams) ---
export default function StudioPage() {
  return (
    <Suspense fallback={
      <div className="h-screen flex items-center justify-center text-muted-foreground text-sm">
        Loading…
      </div>
    }>
      <StudioContent />
    </Suspense>
  )
}
