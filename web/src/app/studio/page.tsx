"use client"

import React, { useState, useRef, useEffect, useMemo, Suspense } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import {
  ArrowLeft, ArrowCounterClockwise, ArrowClockwise,
  Export, Play, Pause, Plus, DotsThreeVertical,
  MinusCircle, PlusCircle, Waveform,
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
import { getSession, type Segment } from "@/lib/session-store"

// --- Constants ---
const SPEAKER_COLORS = [
  { avatar: "bg-blue-400",    track: "bg-blue-200" },
  { avatar: "bg-pink-400",    track: "bg-pink-200" },
  { avatar: "bg-emerald-400", track: "bg-emerald-200" },
  { avatar: "bg-amber-400",   track: "bg-amber-200" },
  { avatar: "bg-violet-400",  track: "bg-violet-200" },
  { avatar: "bg-cyan-400",    track: "bg-cyan-200" },
  { avatar: "bg-rose-400",    track: "bg-rose-200" },
  { avatar: "bg-lime-400",    track: "bg-lime-200" },
]

// --- Mock Data (demo / fallback) ---
const MOCK_SEGMENTS: Segment[] = [
  { id: 1, speaker: "SPEAKER_00", start: 0.10,  end: 7.28,  text: "[instrumental music plays]", emotion: "Neutral", valence: 0.0,  arousal: 0.1 },
  { id: 2, speaker: "SPEAKER_00", start: 8.10,  end: 9.04,  text: "Hello, I'm here.", emotion: "Calm", valence: 0.3,  arousal: -0.1 },
  { id: 3, speaker: "SPEAKER_01", start: 10.62, end: 11.00, text: "Oh.", emotion: "Surprise", valence: 0.4,  arousal: 0.6 },
  { id: 4, speaker: "SPEAKER_01", start: 13.02, end: 14.18, text: "Hi.", emotion: "Neutral", valence: 0.1,  arousal: 0.0 },
  { id: 5, speaker: "SPEAKER_00", start: 14.82, end: 16.40, text: "Hi.", emotion: "Happy", valence: 0.7,  arousal: 0.4 },
  { id: 6, speaker: "SPEAKER_01", start: 17.20, end: 22.10, text: "It's been a long time coming. Are you nervous?", emotion: "Curious", valence: 0.2,  arousal: 0.5 },
  { id: 7, speaker: "SPEAKER_00", start: 22.90, end: 28.50, text: "A little bit. The data is just... it's a lot to process.", emotion: "Anxiety", valence: -0.4, arousal: 0.7 },
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
      return [id, { label: `Speaker ${i + 1}`, avatarColor: palette.avatar, trackColor: palette.track }]
    })
  )
}

// --- SegmentRow ---
function SegmentRow({
  seg,
  speakerMap,
  active,
  onClick,
}: {
  seg: Segment
  speakerMap: Record<string, SpeakerInfo>
  active: boolean
  onClick: () => void
}) {
  const speaker = speakerMap[seg.speaker] ?? { label: seg.speaker, avatarColor: "bg-gray-400", trackColor: "bg-gray-200" }
  return (
    <div
      onClick={onClick}
      className={`flex gap-0 group transition-colors cursor-pointer border-b border-border last:border-0 ${
        active ? "bg-accent" : "hover:bg-muted/40"
      }`}
    >
      {/* Speaker col */}
      <div className="w-36 flex-shrink-0 flex items-start gap-2 px-4 py-4">
        <Avatar className="w-7 h-7 flex-shrink-0 mt-0.5">
          <AvatarFallback className={`${speaker.avatarColor} text-white text-[10px] font-bold`}>
            {speaker.label[0]}
          </AvatarFallback>
        </Avatar>
        <span className="text-xs font-bold text-foreground leading-tight mt-1">{speaker.label}</span>
      </div>

      <Separator orientation="vertical" className="h-auto" />

      {/* Time + text col */}
      <div className="flex-1 px-5 py-4 space-y-1 min-w-0">
        <p className="text-[11px] text-muted-foreground font-medium">{fmtTime(seg.start)}</p>
        <p className="text-sm text-foreground leading-relaxed">{seg.text}</p>
        <p className="text-[11px] text-muted-foreground font-medium">{fmtTime(seg.end)}</p>
      </div>

      {/* Emotion badges */}
      <div className="flex-shrink-0 flex flex-col items-end gap-1 pt-4 pr-4">
        <Badge
          variant="outline"
          className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0 border-border text-muted-foreground"
        >
          {seg.emotion}
        </Badge>
        {seg.face_emotion && (
          <Badge
            variant="outline"
            className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0 border-border text-muted-foreground/60"
          >
            {seg.face_emotion}
          </Badge>
        )}
      </div>
    </div>
  )
}

// --- RightPanel ---
function RightPanel({
  activeSegment,
  audioUrl,
  filename,
  isPlaying,
  isVideo,
  mediaRef,
  currentTime,
  duration,
  onToggle,
}: {
  activeSegment: Segment | null
  audioUrl: string
  filename: string
  isPlaying: boolean
  isVideo: boolean
  mediaRef: React.RefObject<HTMLVideoElement | null>
  currentTime: number
  duration: number
  onToggle: () => void
}) {
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="flex flex-col h-full border-l border-border bg-card">
      {/* Media preview panel */}
      <div className="aspect-video w-full bg-slate-950 flex items-center justify-center flex-shrink-0 relative overflow-hidden">
        {isVideo && audioUrl ? (
          <video
            ref={mediaRef}
            src={audioUrl}
            preload="metadata"
            className="w-full h-full object-contain"
          />
        ) : (
          <>
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-white/20">
              <Waveform size={36} />
              <span className="text-[10px] font-bold uppercase tracking-widest truncate max-w-[80%] text-center px-2">
                {filename}
              </span>
            </div>
            <video ref={mediaRef} src={audioUrl || undefined} preload="metadata" className="hidden" />
          </>
        )}
        {/* Controls bar */}
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-black/70 flex items-center px-2 gap-2">
          <button
            onClick={onToggle}
            disabled={!audioUrl}
            className="text-white/80 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {isPlaying
              ? <Pause size={12} weight="fill" />
              : <Play size={12} weight="fill" />
            }
          </button>
          <div className="flex-1 h-[2px] bg-white/20 rounded-full mx-1">
            <div
              className="h-full bg-white/60 rounded-full transition-[width]"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="text-white/40 text-[9px] font-mono">{fmtTime(currentTime)}</span>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Global Properties */}
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase tracking-widest hover:text-foreground transition-colors w-full text-left">
              <span>▾</span> Global Properties
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-3 space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-foreground truncate max-w-[160px] text-xs">{filename}</span>
                </div>
                <Separator />
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs">Duration</span>
                  <span className="font-semibold text-xs">{fmtTime(duration)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs">Subtitles</span>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-6 w-6">
                        <DotsThreeVertical size={14} />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent>
                      <DropdownMenuItem>Export SRT</DropdownMenuItem>
                      <DropdownMenuItem>Export VTT</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          <Separator />

          {/* Emotional Analysis */}
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase tracking-widest hover:text-foreground transition-colors w-full text-left">
              <span>▾</span> Emotional Analysis
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-3 space-y-3">
                {activeSegment ? (
                  <>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Voice Emotion</span>
                      <Badge variant="outline" className="font-bold text-[10px]">{activeSegment.emotion}</Badge>
                    </div>
                    {activeSegment.face_emotion && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Face Emotion</span>
                        <Badge variant="outline" className="font-bold text-[10px] text-muted-foreground/70">{activeSegment.face_emotion}</Badge>
                      </div>
                    )}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span>Valence</span>
                        <span className="font-bold text-foreground">{((activeSegment.valence + 1) / 2 * 100).toFixed(0)}%</span>
                      </div>
                      <Progress value={(activeSegment.valence + 1) / 2 * 100} className="h-1.5" />
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span>Arousal</span>
                        <span className="font-bold text-foreground">{((activeSegment.arousal + 1) / 2 * 100).toFixed(0)}%</span>
                      </div>
                      <Progress value={(activeSegment.arousal + 1) / 2 * 100} className="h-1.5" />
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">Select a segment to view analysis.</p>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
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
  speakerMap,
  duration,
  currentTime,
}: {
  isPlaying: boolean
  onToggle: () => void
  segments: Segment[]
  speakerMap: Record<string, SpeakerInfo>
  duration: number
  currentTime: number
}) {
  const speakers = Object.keys(speakerMap)

  return (
    <div className="border-t border-border bg-card flex-shrink-0">
      {/* Speaker track rows */}
      <div className="border-b border-border">
        {speakers.map((speakerId) => {
          const info = speakerMap[speakerId]
          const speakerSegs = segments.filter(s => s.speaker === speakerId)

          return (
            <div key={speakerId} className="flex items-center px-3 py-1.5 gap-2 group">
              <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground cursor-grab">
                <DotsThreeVertical size={12} />
              </Button>
              <Avatar className="w-5 h-5">
                <AvatarFallback className={`${info.avatarColor} text-white text-[8px] font-bold`}>S</AvatarFallback>
              </Avatar>
              <span className="text-xs font-semibold text-foreground w-20">{info.label}</span>
              <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground ml-auto opacity-0 group-hover:opacity-100">
                <DotsThreeVertical size={12} />
              </Button>
              {/* Track visualization */}
              <div className="flex-1 h-5 relative overflow-hidden rounded-sm bg-muted/30">
                {duration > 0 && speakerSegs.map(seg => (
                  <div
                    key={seg.id}
                    className={`absolute top-0.5 bottom-0.5 ${info.trackColor} rounded-sm`}
                    style={{
                      left: `${(seg.start / duration) * 100}%`,
                      width: `${Math.max(((seg.end - seg.start) / duration) * 100, 0.5)}%`,
                    }}
                  />
                ))}
                {/* Playhead */}
                {duration > 0 && (
                  <div
                    className="absolute top-0 bottom-0 w-px bg-foreground/60 z-10"
                    style={{ left: `${(currentTime / duration) * 100}%` }}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Playback controls */}
      <div className="flex items-center gap-4 px-4 h-10">
        {/* Zoom */}
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <MinusCircle size={14} />
          <Slider defaultValue={[40]} max={100} className="w-16" />
          <PlusCircle size={14} />
        </div>

        <Separator orientation="vertical" className="h-5" />

        {/* Play / speed */}
        <div className="flex items-center gap-3">
          <Button
            size="icon"
            className="h-7 w-7 rounded-full"
            onClick={onToggle}
          >
            {isPlaying
              ? <Pause size={13} weight="fill" />
              : <Play size={13} weight="fill" />
            }
          </Button>
          <span className="text-xs font-bold text-muted-foreground">1.0×</span>
        </div>

        <div className="flex-1" />

        <Button variant="ghost" size="sm" className="text-xs text-muted-foreground font-semibold gap-1.5 h-7">
          <Plus size={13} />
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

  const mediaRef = useRef<HTMLVideoElement>(null)
  const [activeId, setActiveId] = useState<number>(1)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)

  const session = sessionId ? getSession(sessionId) : null
  const segments = session?.data.segments ?? MOCK_SEGMENTS
  const audioUrl = session?.audioUrl ?? ""
  const filename = session?.filename ?? MOCK_FILENAME
  const duration = session?.data.duration ?? MOCK_DURATION
  const isVideo = session?.data.has_video ?? false

  const speakerMap = useMemo(() => buildSpeakerMap(segments), [segments])
  const activeSegment = segments.find(s => s.id === activeId) ?? segments[0] ?? null

  useEffect(() => {
    const media = mediaRef.current
    if (!media) return
    const onTime = () => setCurrentTime(media.currentTime)
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    media.addEventListener("timeupdate", onTime)
    media.addEventListener("play", onPlay)
    media.addEventListener("pause", onPause)
    return () => {
      media.removeEventListener("timeupdate", onTime)
      media.removeEventListener("play", onPlay)
      media.removeEventListener("pause", onPause)
    }
  }, [])

  const handleSegmentClick = (seg: Segment) => {
    setActiveId(seg.id)
    if (mediaRef.current && audioUrl) {
      mediaRef.current.currentTime = seg.start
    }
  }

  const togglePlay = () => {
    const media = mediaRef.current
    if (!media || !audioUrl) return
    if (isPlaying) media.pause()
    else media.play()
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Top Bar */}
      <header className="h-11 border-b border-border bg-card flex items-center px-4 gap-3 flex-shrink-0 sticky top-0 z-50">
        <Link href="/">
          <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-foreground">
            <ArrowLeft size={16} />
          </Button>
        </Link>

        <div className="flex items-center gap-0.5 ml-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground">
                <ArrowCounterClockwise size={15} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Undo</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground">
                <ArrowClockwise size={15} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Redo</TooltipContent>
          </Tooltip>
        </div>

        <div className="flex-1" />

        <Button className="gap-1.5 font-bold h-8 text-xs px-3">
          <Export size={14} weight="bold" />
          Export
        </Button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Transcript */}
        <div className="flex-1 overflow-hidden flex flex-col min-w-0">
          <ScrollArea className="flex-1">
            <div className="max-w-2xl mx-auto py-4">
              {segments.map(seg => (
                <SegmentRow
                  key={seg.id}
                  seg={seg}
                  speakerMap={speakerMap}
                  active={seg.id === activeId}
                  onClick={() => handleSegmentClick(seg)}
                />
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Right: Properties panel */}
        <div className="w-[280px] flex-shrink-0 overflow-hidden flex flex-col">
          <RightPanel
            activeSegment={activeSegment}
            audioUrl={audioUrl}
            filename={filename}
            isPlaying={isPlaying}
            isVideo={isVideo}
            mediaRef={mediaRef}
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
        speakerMap={speakerMap}
        duration={duration}
        currentTime={currentTime}
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
