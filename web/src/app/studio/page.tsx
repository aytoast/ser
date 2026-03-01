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
  state,
  speaker,
  onRowClick,
  onSeek,
  containerRef,
  currentTime,
}: {
  seg: Segment
  state: "past" | "active" | "future"
  speaker: SpeakerInfo
  onRowClick: () => void
  onSeek: (time: number) => void
  containerRef?: (el: HTMLDivElement | null) => void
  currentTime: number
}) {
  const active = state === "active"
  const segDuration = Math.max(seg.end - seg.start, 0.001)

  // Strip [bracket] tags from text for character counting/highlighting
  const cleanText = useMemo(() => seg.text.replace(/\[[^\]]+\]/g, "").trim(), [seg.text])

  // Split text into Unicode-aware characters (handles CJK, emoji, etc.)
  const chars = useMemo(() => Array.from(cleanText), [cleanText])

  // Parse text into ordered tokens: plain text chunks and [bracket] tags at correct positions.
  // Tag tokens carry cleanOffset = number of clean chars before them (used for badge click-to-seek).
  const tokens = useMemo(() => {
    type Token = { type: "text"; content: string } | { type: "tag"; content: string; cleanOffset: number }
    const result: Token[] = []
    let lastIndex = 0
    let cleanCharCount = 0
    const re = /\[([^\]]+)\]/g
    let m
    while ((m = re.exec(seg.text)) !== null) {
      if (m.index > lastIndex) {
        const chunk = seg.text.slice(lastIndex, m.index)
        result.push({ type: "text", content: chunk })
        cleanCharCount += Array.from(chunk).length
      }
      result.push({ type: "tag", content: m[1], cleanOffset: cleanCharCount })
      lastIndex = m.index + m[0].length
    }
    if (lastIndex < seg.text.length) result.push({ type: "text", content: seg.text.slice(lastIndex) })
    return result
  }, [seg.text])

  // How many chars are "lit" (already played through)
  const litCount = state === "past"
    ? chars.length
    : active
    ? Math.floor(Math.max(0, currentTime - seg.start) / segDuration * chars.length)
    : 0

  // Click on text → find char index via caret position → seek to proportional time
  // Badge text nodes are excluded from char counting via data-badge filter
  function handleTextClick(e: React.MouseEvent<HTMLParagraphElement>) {
    e.stopPropagation()
    let charIndex = litCount

    if (typeof document.caretRangeFromPoint === "function") {
      const range = document.caretRangeFromPoint(e.clientX, e.clientY)
      if (range) {
        let offset = range.startOffset
        const walker = document.createTreeWalker(e.currentTarget, NodeFilter.SHOW_TEXT, {
          acceptNode: (node) => {
            // Skip text nodes inside [data-badge] elements
            let el = node.parentElement
            while (el && el !== e.currentTarget) {
              if ((el as HTMLElement).dataset?.badge) return NodeFilter.FILTER_SKIP
              el = el.parentElement
            }
            return NodeFilter.FILTER_ACCEPT
          },
        })
        let node = walker.nextNode()
        while (node && node !== range.startContainer) {
          offset += node.textContent?.length ?? 0
          node = walker.nextNode()
        }
        charIndex = Math.max(0, Math.min(chars.length - 1, offset))
      }
    } else {
      // Fallback: proportional X within paragraph
      const rect = e.currentTarget.getBoundingClientRect()
      charIndex = Math.floor(((e.clientX - rect.left) / rect.width) * chars.length)
    }

    const seekTime = seg.start + (charIndex / chars.length) * segDuration
    onSeek(Math.max(seg.start, Math.min(seg.end, seekTime)))
  }

  return (
    <div
      ref={containerRef}
      onClick={onRowClick}
      className={cn(
        "relative flex items-start gap-12 group transition-colors duration-200 cursor-pointer py-4 rounded-lg px-4 border border-transparent",
        !active && "hover:bg-muted/20"
      )}
    >
      {/* Speaker */}
      <div className={cn(
        "w-32 flex items-center gap-3 shrink-0 pt-1 transition-opacity duration-300",
        state === "past" ? "opacity-30" : state === "future" ? "opacity-50" : "opacity-100"
      )}>
        <Avatar className="size-6 ring-2 ring-background border border-border/20">
          <AvatarFallback className={cn("text-[10px] text-white", speaker.avatarColor)}>
            {speaker.label[0]}
          </AvatarFallback>
        </Avatar>
        <span className="text-[13px] font-semibold text-foreground/70">{speaker.label}</span>
      </div>

      {/* Content */}
      <div className={cn(
        "flex-1 space-y-1.5 min-w-0 pl-8 relative border-l-[1.5px] transition-colors duration-300",
        active ? "border-primary/40" : state === "past" ? "border-border/15" : "border-border/25"
      )}>
        <span className="text-[10px] font-sans font-medium text-muted-foreground/40 block tracking-tight">{fmtTime(seg.start)}</span>

        {/* Tokenized text: text chunks lit/dim + [bracket] badges at correct inline positions */}
        <p
          className="text-[15px] leading-[1.6] font-medium tracking-tight whitespace-pre-wrap cursor-text select-none"
          onClick={handleTextClick}
        >
          {(() => {
            let cleanOffset = 0
            return tokens.map((token, i) => {
              if (token.type === "tag") {
                const seekToTag = (e: React.MouseEvent) => {
                  e.stopPropagation()
                  if (chars.length === 0) return
                  const seekTime = seg.start + (token.cleanOffset / chars.length) * segDuration
                  onSeek(Math.max(seg.start, Math.min(seg.end, seekTime)))
                }
                return (
                  <span key={i} data-badge="true" className="inline-flex items-center mx-1 align-middle cursor-pointer" onClick={seekToTag}>
                    <Badge
                      variant="secondary"
                      className={cn("text-[10px] h-5 px-2 font-medium rounded-full transition-opacity duration-300 hover:bg-secondary/80", state === "past" && "opacity-40")}
                    >
                      {token.content}
                    </Badge>
                  </span>
                )
              }
              // Text token — split into lit + dim based on global cleanOffset
              const tokenChars = Array.from(token.content)
              const tokenLen = tokenChars.length
              const localLit = Math.max(0, Math.min(tokenLen, litCount - cleanOffset))
              cleanOffset += tokenLen
              return (
                <React.Fragment key={i}>
                  <span className="text-foreground">{tokenChars.slice(0, localLit).join("")}</span>
                  <span className="text-muted-foreground/30">{tokenChars.slice(localLit).join("")}</span>
                </React.Fragment>
              )
            })
          })()}
        </p>
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

const VIDEO_EXTS = new Set([".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm"])

function isVideoFile(filename: string): boolean {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase()
  return VIDEO_EXTS.has(ext)
}

// Mirrors Python _TAG_EMOTIONS in api/main.py — bracket tag → { emotion, valence, arousal }
const TAG_EMOTIONS: Record<string, { emotion: string; valence: number; arousal: number }> = {
  "laughs":           { emotion: "Happy",       valence:  0.70, arousal:  0.60 },
  "laughing":         { emotion: "Happy",       valence:  0.70, arousal:  0.60 },
  "chuckles":         { emotion: "Happy",       valence:  0.50, arousal:  0.30 },
  "giggles":          { emotion: "Happy",       valence:  0.60, arousal:  0.40 },
  "sighs":            { emotion: "Sad",         valence: -0.30, arousal: -0.30 },
  "sighing":          { emotion: "Sad",         valence: -0.30, arousal: -0.30 },
  "cries":            { emotion: "Sad",         valence: -0.70, arousal:  0.40 },
  "crying":           { emotion: "Sad",         valence: -0.70, arousal:  0.40 },
  "whispers":         { emotion: "Calm",        valence:  0.10, arousal: -0.50 },
  "whispering":       { emotion: "Calm",        valence:  0.10, arousal: -0.50 },
  "shouts":           { emotion: "Angry",       valence: -0.50, arousal:  0.80 },
  "shouting":         { emotion: "Angry",       valence: -0.50, arousal:  0.80 },
  "exclaims":         { emotion: "Excited",     valence:  0.50, arousal:  0.70 },
  "gasps":            { emotion: "Surprised",   valence:  0.20, arousal:  0.70 },
  "hesitates":        { emotion: "Anxious",     valence: -0.20, arousal:  0.30 },
  "stutters":         { emotion: "Anxious",     valence: -0.20, arousal:  0.40 },
  "stammers":         { emotion: "Anxious",     valence: -0.25, arousal:  0.35 },
  "mumbles":          { emotion: "Sad",         valence: -0.20, arousal: -0.30 },
  "nervous":          { emotion: "Anxious",     valence: -0.30, arousal:  0.40 },
  "frustrated":       { emotion: "Frustrated",  valence: -0.50, arousal:  0.50 },
  "excited":          { emotion: "Excited",     valence:  0.50, arousal:  0.70 },
  "sad":              { emotion: "Sad",         valence: -0.60, arousal: -0.20 },
  "angry":            { emotion: "Angry",       valence: -0.60, arousal:  0.70 },
  "claps":            { emotion: "Happy",       valence:  0.60, arousal:  0.50 },
  "applause":         { emotion: "Happy",       valence:  0.60, arousal:  0.50 },
  "clears throat":    { emotion: "Neutral",     valence:  0.00, arousal:  0.10 },
  "pause":            { emotion: "Neutral",     valence:  0.00, arousal: -0.10 },
  "laughs nervously": { emotion: "Anxious",     valence: -0.10, arousal:  0.40 },
}

function getTagEntry(tag: string) {
  const lower = tag.toLowerCase().trim()
  if (TAG_EMOTIONS[lower]) return TAG_EMOTIONS[lower]
  for (const [key, entry] of Object.entries(TAG_EMOTIONS)) {
    if (lower.includes(key)) return entry
  }
  return null
}

// --- RightPanel ---
function RightPanel({
  filename,
  audioUrl,
  isVideo,
  mediaRef,
  segments,
  faceTimeline,
  currentTime,
  isPlaying,
  onToggle,
}: {
  activeSegment: Segment | null
  audioUrl: string
  filename: string
  isVideo: boolean
  mediaRef: React.RefObject<HTMLVideoElement | null>
  segments: Segment[]
  faceTimeline: Record<string, string>
  isPlaying: boolean
  currentTime: number
  duration: number
  onToggle: () => void
}) {
  // Find the segment active at the current playback position
  const liveSeg = segments.find(s => currentTime >= s.start && currentTime < s.end) ?? null

  // Per-second face emotion from timeline (more granular than segment majority-vote)
  const liveFaceEmo = faceTimeline[String(Math.floor(currentTime))] ?? liveSeg?.face_emotion ?? null

  // Streaming speech emotion + valence + arousal — derived from [bracket] tag positions.
  // Each tag's timing is estimated proportionally by clean-char position within the segment.
  const liveSpeech = useMemo(() => {
    if (!liveSeg) return null
    const text = liveSeg.text
    const segDuration = Math.max(liveSeg.end - liveSeg.start, 0.001)
    const cleanText = text.replace(/\[[^\]]+\]/g, "").trim()
    const totalChars = Array.from(cleanText).length

    let last: { emotion: string; valence: number; arousal: number } | null = null
    if (totalChars > 0) {
      let cleanCharsBefore = 0
      let rawPos = 0
      const re = /\[([^\]]+)\]/g
      let m
      while ((m = re.exec(text)) !== null) {
        const chunkBefore = text.slice(rawPos, m.index).replace(/\[[^\]]+\]/g, "")
        cleanCharsBefore += Array.from(chunkBefore).length
        rawPos = m.index + m[0].length
        const tagTime = liveSeg.start + (cleanCharsBefore / totalChars) * segDuration
        if (tagTime <= currentTime) {
          const entry = getTagEntry(m[1])
          if (entry) last = entry
        }
      }
    }

    return {
      emo:     last?.emotion ?? liveSeg.emotion,
      valence: last?.valence ?? liveSeg.valence,
      arousal: last?.arousal ?? liveSeg.arousal,
    }
  }, [liveSeg, currentTime])

  return (
    <div className="flex flex-col h-full border-l border-border bg-background">
      {/* Video / Audio Preview */}
      <div className="aspect-video w-full bg-slate-950 flex items-center justify-center flex-shrink-0 relative overflow-hidden">
        <video
          ref={mediaRef as React.RefObject<HTMLVideoElement>}
          src={audioUrl || undefined}
          preload="metadata"
          className={isVideo && audioUrl ? "w-full h-full object-contain" : "hidden"}
        />
        {(!isVideo || !audioUrl) && (
          <NextImage src="/logo.svg" alt="Preview" width={48} height={48} className="opacity-10" />
        )}
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

          {/* Live Emotion — streams with playback */}
          {segments.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground uppercase tracking-widest">
                <span className="text-[8px]">▼</span>
                <span>Emotion</span>
                {isPlaying && (
                  <span className="ml-auto flex items-center gap-1 text-[10px] font-normal normal-case text-emerald-500">
                    <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Live
                  </span>
                )}
              </div>

              {liveSeg && liveSpeech ? (
                <div className="space-y-4 pt-1">
                  {/* Speech emotion — streams sub-segment via bracket tag timing */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Speech</span>
                    <Badge key={liveSpeech.emo} variant="secondary" className="text-[11px] h-5 px-2 font-medium rounded-full transition-all duration-300 animate-in fade-in zoom-in-95 duration-200">
                      {liveSpeech.emo}
                    </Badge>
                  </div>

                  {/* Face emotion — per-second from timeline, fallback to segment majority */}
                  {liveFaceEmo && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Face</span>
                      <Badge variant="outline" className="text-[11px] h-5 px-2 font-medium rounded-full gap-1 transition-all duration-300">
                        <VideoCamera size={9} weight="fill" className="text-pink-500" />
                        {liveFaceEmo}
                      </Badge>
                    </div>
                  )}

                  {/* Valence bar — streams with bracket tag valence */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>Valence</span>
                      <span className={liveSpeech.valence >= 0 ? "text-emerald-500" : "text-red-400"}>
                        {liveSpeech.valence > 0 ? "+" : ""}{liveSpeech.valence.toFixed(2)}
                      </span>
                    </div>
                    <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                      <div className="absolute top-0 left-1/2 h-full w-px bg-border/60 z-10" />
                      <div
                        className={cn(
                          "absolute top-0 h-full rounded-full transition-all duration-500",
                          liveSpeech.valence >= 0 ? "bg-emerald-400" : "bg-red-400"
                        )}
                        style={{
                          left: liveSpeech.valence >= 0 ? "50%" : `${(0.5 + liveSpeech.valence / 2) * 100}%`,
                          width: `${Math.abs(liveSpeech.valence) * 50}%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[9px] text-muted-foreground/40">
                      <span>Negative</span><span>Positive</span>
                    </div>
                  </div>

                  {/* Arousal bar — streams with bracket tag arousal */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>Arousal</span>
                      <span className={liveSpeech.arousal >= 0 ? "text-blue-400" : "text-slate-400"}>
                        {liveSpeech.arousal > 0 ? "+" : ""}{liveSpeech.arousal.toFixed(2)}
                      </span>
                    </div>
                    <div className="relative h-1.5 bg-muted rounded-full overflow-hidden">
                      <div className="absolute top-0 left-1/2 h-full w-px bg-border/60 z-10" />
                      <div
                        className={cn(
                          "absolute top-0 h-full rounded-full transition-all duration-500",
                          liveSpeech.arousal >= 0 ? "bg-blue-400" : "bg-slate-400"
                        )}
                        style={{
                          left: liveSpeech.arousal >= 0 ? "50%" : `${(0.5 + liveSpeech.arousal / 2) * 100}%`,
                          width: `${Math.abs(liveSpeech.arousal) * 50}%`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[9px] text-muted-foreground/40">
                      <span>Calm</span><span>Excited</span>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-[11px] text-muted-foreground/40 text-center py-3">
                  {currentTime === 0 ? "Play to see live emotion" : "—"}
                </p>
              )}
            </div>
          )}

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
  activeSegId,
  onSeek,
}: {
  isPlaying: boolean
  onToggle: () => void
  segments: Segment[]
  duration: number
  currentTime: number
  speakerMap: Record<string, SpeakerInfo>
  activeSegId: number
  onSeek: (time: number) => void
}) {
  const speakers = Object.entries(speakerMap)

  function handleTracksClick(e: React.MouseEvent<HTMLDivElement>) {
    if (duration <= 0) return
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    onSeek(ratio * duration)
  }

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

        {/* Tracks Area — click anywhere to seek */}
        <div className="flex-1 relative overflow-hidden cursor-pointer" onClick={handleTracksClick}>
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
                    className={cn(
                      "absolute top-2 bottom-2 rounded-[6px] transition-all duration-150",
                      info.trackColor,
                      seg.id === activeSegId
                        ? "opacity-100 ring-1 ring-foreground/30 ring-inset"
                        : "opacity-70 hover:opacity-90"
                    )}
                    style={{
                      left: `${(seg.start / duration) * 100}%`,
                      width: `${Math.max(((seg.end - seg.start) / duration) * 100, 0.5)}%`,
                    }}
                  />
                ))}
              </div>
            ))}

            {/* Played region overlay */}
            {duration > 0 && currentTime > 0 && (
              <div
                className="absolute inset-y-0 left-0 bg-foreground/[0.04] pointer-events-none"
                style={{ width: (currentTime / duration) * 100 + "%" }}
              />
            )}

            {/* Playhead — thin line + dot at top */}
            {duration > 0 && (
              <div
                className="absolute top-0 bottom-0 z-20 pointer-events-none"
                style={{ left: `calc(${(currentTime / duration) * 100}% - 1px)` }}
              >
                <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 size-2.5 rounded-full bg-foreground border-2 border-background" />
                <div className="absolute top-1 bottom-0 left-1/2 -translate-x-1/2 w-[1.5px] bg-foreground/80" />
              </div>
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

  const mediaRef = useRef<HTMLVideoElement>(null)
  // Ref-based guard: set synchronously before any await to prevent double-fetch
  // even when React re-runs the effect before setIsProcessing(true) is committed.
  const processingRef = useRef(false)
  // Per-segment DOM element refs for auto-scroll
  const segmentRefs = useRef<Map<number, HTMLDivElement>>(new Map())

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

  // Automatic processing for pending sessions.
  // Uses async job polling: POST returns {job_id} immediately, then GET /api/job/:id
  // until done — avoids HF Spaces ~3 min proxy timeout during long CPU inference.
  useEffect(() => {
    if (!session || processingRef.current || processError) return

    if (session.file && session.data.segments.length === 0) {
      processingRef.current = true
      const process = async () => {
        setIsProcessing(true)
        setProcessError(null)
        try {
          // 1. Submit job — server responds immediately with job_id (202)
          const formData = new FormData()
          formData.append("audio", session.file!, session.filename)

          const submitRes = await fetch(`${API_BASE}/api/transcribe-diarize`, {
            method: "POST",
            body: formData,
          })

          if (!submitRes.ok) {
            const errData = await submitRes.json().catch(() => ({}))
            throw new Error(errData.error ?? "Submit failed")
          }

          const { job_id } = await submitRes.json() as { job_id: string }

          // 2. Poll until done (every 3s)
          const POLL_INTERVAL = 3000
          const MAX_POLLS = 60 * 20  // 60 min max
          let polls = 0

          const data = await new Promise<DiarizeResult>((resolve, reject) => {
            const tick = async () => {
              polls++
              if (polls > MAX_POLLS) {
                reject(new Error("Processing timed out after 60 minutes"))
                return
              }
              try {
                const pollRes = await fetch(`${API_BASE}/api/job/${job_id}`)
                const pollData = await pollRes.json()
                if (pollData.status === "done") {
                  resolve(pollData.data as DiarizeResult)
                } else if (pollData.status === "error") {
                  reject(new Error(pollData.error ?? "Processing failed"))
                } else {
                  // still pending — keep polling
                  setTimeout(tick, POLL_INTERVAL)
                }
              } catch (e) {
                reject(e)
              }
            }
            setTimeout(tick, POLL_INTERVAL)
          })

          updateSession(session.id, data)
          const updated = getSession(session.id)
          setSession(updated)
          if (updated?.data.segments && updated.data.segments.length > 0) {
            setActiveId(updated.data.segments[0].id)
          }
        } catch (e) {
          processingRef.current = false
          setProcessError(e instanceof Error ? e.message : "Request failed")
        } finally {
          setIsProcessing(false)
        }
      }
      process()
    }
  }, [session, processError])

  const segments = session?.data.segments ?? (session ? [] : MOCK_SEGMENTS)
  const audioUrl = session?.audioUrl ?? ""
  const filename = session?.filename ?? MOCK_FILENAME
  const duration = session?.data.duration ?? MOCK_DURATION
  const faceTimeline = session?.data.face_emotion_timeline ?? {}

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
  }, [audioUrl])

  const isVideo = isVideoFile(filename)

  // Update activeId based on current playback time
  useEffect(() => {
    if (!isPlaying) return
    const seg = segments.find(s => currentTime >= s.start && currentTime < s.end)
    if (seg && seg.id !== activeId) {
      setActiveId(seg.id)
    }
  }, [currentTime, segments, isPlaying, activeId])

  // Auto-scroll transcript to active segment whenever activeId changes
  useEffect(() => {
    const el = segmentRefs.current.get(activeId)
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [activeId])

  const handleSeek = (time: number) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = time
      setCurrentTime(time)
    }
    // Immediately highlight the segment at the seeked time
    const seg = segments.find(s => time >= s.start && time < s.end)
    if (seg) setActiveId(seg.id)
  }

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

          <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
            <div className="max-w-none px-[200px] py-10">
              {segments.length === 0 && !isProcessing && !processError && (
                <div className="py-20 text-center text-muted-foreground">
                  No segments found.
                </div>
              )}
              {segments.map((seg, i) => {
                const state = seg.id === activeId ? "active"
                  : seg.end <= currentTime ? "past"
                  : "future"
                return (
                  <React.Fragment key={seg.id}>
                    {i > 0 && segments[i - 1].speaker === seg.speaker && <MergeButton />}
                    <SegmentRow
                      seg={seg}
                      state={state}
                      speaker={speakerMap[seg.speaker]}
                      onRowClick={() => handleSegmentClick(seg)}
                      onSeek={handleSeek}
                      currentTime={currentTime}
                      containerRef={el => {
                        if (el) segmentRefs.current.set(seg.id, el)
                        else segmentRefs.current.delete(seg.id)
                      }}
                    />
                  </React.Fragment>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right: Properties panel */}
        <div className="w-[320px] flex-shrink-0 overflow-hidden flex flex-col">
          <RightPanel
            activeSegment={activeSegment}
            audioUrl={audioUrl}
            filename={filename}
            isVideo={isVideo}
            mediaRef={mediaRef}
            segments={segments}
            faceTimeline={faceTimeline}
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
        activeSegId={activeId}
        onSeek={handleSeek}
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
