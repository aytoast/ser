"use client"

import React, { useState } from "react"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  ArrowLeft, ArrowCounterClockwise, ArrowClockwise,
  Export, Play, Pause, Plus, DotsThreeVertical,
  MagnifyingGlass, MinusCircle, PlusCircle, Waveform,
  SpeakerHigh,
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

// --- Mock Data ---
const SPEAKERS = [
  { id: "s0", label: "Speaker 0", color: "bg-blue-400" },
  { id: "s1", label: "Speaker 1", color: "bg-pink-400" },
]

const SEGMENTS = [
  { id: 1, speaker: "s0", startTime: "00.10", endTime: "07.28", text: "[instrumental music plays]", emotion: "Neutral", valence: 0.0, arousal: 0.1 },
  { id: 2, speaker: "s0", startTime: "08.10", endTime: "09.04", text: "Hello, I'm here.", emotion: "Calm", valence: 0.3, arousal: -0.1 },
  { id: 3, speaker: "s1", startTime: "10.62", endTime: "11.00", text: "Oh.", emotion: "Surprise", valence: 0.4, arousal: 0.6 },
  { id: 4, speaker: "s1", startTime: "13.02", endTime: "14.18", text: "Hi.", emotion: "Neutral", valence: 0.1, arousal: 0.0 },
  { id: 5, speaker: "s0", startTime: "14.82", endTime: "16.40", text: "Hi.", emotion: "Happy", valence: 0.7, arousal: 0.4 },
  { id: 6, speaker: "s1", startTime: "17.20", endTime: "22.10", text: "It's been a long time coming. Are you nervous?", emotion: "Curious", valence: 0.2, arousal: 0.5 },
  { id: 7, speaker: "s0", startTime: "22.90", endTime: "28.50", text: "A little bit. The data is just... it's a lot to process.", emotion: "Anxiety", valence: -0.4, arousal: 0.7 },
]

const SPEAKER_MAP: Record<string, { label: string; color: string }> = {
  s0: { label: "Speaker 0", color: "bg-blue-400" },
  s1: { label: "Speaker 1", color: "bg-pink-400" },
}

// Transcript segment row — matches the reference's Speaker / Timestamp / Text layout
function SegmentRow({
  seg,
  active,
  onClick,
}: {
  seg: typeof SEGMENTS[number]
  active: boolean
  onClick: () => void
}) {
  const speaker = SPEAKER_MAP[seg.speaker]
  return (
    <div
      onClick={onClick}
      className={`flex gap-0 group transition-colors cursor-pointer border-b border-border last:border-0 ${active ? "bg-accent" : "hover:bg-muted/40"}`}
    >
      {/* Speaker col */}
      <div className="w-36 flex-shrink-0 flex items-start gap-2 px-4 py-4">
        <Avatar className="w-7 h-7 flex-shrink-0 mt-0.5">
          <AvatarFallback className={`${speaker.color} text-white text-[10px] font-bold`}>
            {speaker.label[0]}
          </AvatarFallback>
        </Avatar>
        <span className="text-xs font-bold text-foreground leading-tight mt-1">{speaker.label}</span>
      </div>

      {/* Separator */}
      <Separator orientation="vertical" className="h-auto" />

      {/* Time + text col */}
      <div className="flex-1 px-5 py-4 space-y-1 min-w-0">
        <p className="text-[11px] text-muted-foreground font-medium">{seg.startTime}</p>
        <p className="text-sm text-foreground leading-relaxed">{seg.text}</p>
        <p className="text-[11px] text-muted-foreground font-medium">{seg.endTime}</p>
      </div>

      {/* Emotion badge — right */}
      <div className="flex-shrink-0 flex items-start pt-4 pr-4">
        <Badge
          variant="outline"
          className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0 border-border text-muted-foreground"
        >
          {seg.emotion}
        </Badge>
      </div>
    </div>
  )
}

// Right panel: video preview + properties
function RightPanel({ activeSegment }: { activeSegment: typeof SEGMENTS[number] | null }) {
  return (
    <div className="flex flex-col h-full border-l border-border bg-card">
      {/* Fake video player */}
      <div className="aspect-video w-full bg-slate-950 flex items-center justify-center flex-shrink-0 relative">
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-white/20">
          <Waveform size={36} />
          <span className="text-[10px] font-bold uppercase tracking-widest">VIDEO PREVIEW</span>
        </div>
        {/* Fake video controls bar */}
        <div className="absolute bottom-0 left-0 right-0 h-6 bg-black/60 flex items-center px-2 gap-1">
          <Play size={10} weight="fill" className="text-white/60" />
          <div className="flex-1 h-[2px] bg-white/20 rounded-full mx-1">
            <div className="w-1/4 h-full bg-white/60 rounded-full" />
          </div>
          <span className="text-white/40 text-[9px]">0:14.82</span>
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
                  <span className="font-semibold text-foreground truncate max-w-[160px] text-xs">WeChat_20250804025710.mp4</span>
                </div>
                <Separator />
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs">Language</span>
                  <span className="font-semibold text-xs">English</span>
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

          {/* Emotional Analysis (active segment) */}
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="flex items-center gap-2 text-xs font-bold text-muted-foreground uppercase tracking-widest hover:text-foreground transition-colors w-full text-left">
              <span>▾</span> Emotional Analysis
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-3 space-y-3">
                {activeSegment ? (
                  <>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Emotion</span>
                      <Badge variant="outline" className="font-bold text-[10px]">{activeSegment.emotion}</Badge>
                    </div>
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

// Bottom timeline bar
function TimelineBar({ isPlaying, onToggle }: { isPlaying: boolean; onToggle: () => void }) {
  return (
    <div className="border-t border-border bg-card flex-shrink-0">
      {/* Speaker track rows */}
      <div className="border-b border-border">
        <div className="flex items-center px-3 py-1.5 gap-2 group">
          <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground cursor-grab">
            <DotsThreeVertical size={12} />
          </Button>
          <Avatar className="w-5 h-5">
            <AvatarFallback className="bg-blue-400 text-white text-[8px] font-bold">S</AvatarFallback>
          </Avatar>
          <span className="text-xs font-semibold text-foreground w-20">Speaker 0</span>
          <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground ml-auto opacity-0 group-hover:opacity-100">
            <DotsThreeVertical size={12} />
          </Button>
          {/* Track visualization */}
          <div className="flex-1 h-5 flex items-center gap-[2px]">
            {[60, 15, 12, 8, 10, 14].map((w, i) => (
              <div key={i} className="h-4 bg-blue-200 rounded-sm" style={{ width: `${w}%` }} />
            ))}
          </div>
        </div>
        <div className="flex items-center px-3 py-1.5 gap-2 group">
          <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground cursor-grab">
            <DotsThreeVertical size={12} />
          </Button>
          <Avatar className="w-5 h-5">
            <AvatarFallback className="bg-pink-400 text-white text-[8px] font-bold">S</AvatarFallback>
          </Avatar>
          <span className="text-xs font-semibold text-foreground w-20">Speaker 1</span>
          <Button variant="ghost" size="icon" className="h-5 w-5 text-muted-foreground ml-auto opacity-0 group-hover:opacity-100">
            <DotsThreeVertical size={12} />
          </Button>
          <div className="flex-1 h-5 flex items-center gap-[2px]">
            {[0, 0, 5, 6, 0, 8, 5, 6].map((w, i) => (
              w > 0 ? <div key={i} className="h-4 bg-pink-200 rounded-sm" style={{ width: `${w}%` }} /> : <div key={i} style={{ width: "8%" }} />
            ))}
          </div>
        </div>
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

        {/* Add segment */}
        <Button variant="ghost" size="sm" className="text-xs text-muted-foreground font-semibold gap-1.5 h-7">
          <Plus size={13} />
          Add segment
        </Button>
      </div>
    </div>
  )
}

// --- Page ---
export default function StudioPage() {
  const [activeId, setActiveId] = useState<number>(1)
  const [isPlaying, setIsPlaying] = useState(false)

  const activeSegment = SEGMENTS.find(s => s.id === activeId) ?? null

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {/* Top Bar */}
      <header className="h-11 border-b border-border bg-card flex items-center px-4 gap-3 flex-shrink-0 sticky top-0 z-50">
        <Link href="/">
          <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-foreground">
            <ArrowLeft size={16} />
          </Button>
        </Link>


        {/* Undo / Redo */}
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
        {/* Left: Transcript scroll */}
        <div className="flex-1 overflow-hidden flex flex-col min-w-0">
          <ScrollArea className="flex-1">
            <div className="max-w-2xl mx-auto py-4">
              {SEGMENTS.map(seg => (
                <SegmentRow
                  key={seg.id}
                  seg={seg}
                  active={seg.id === activeId}
                  onClick={() => setActiveId(seg.id)}
                />
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Right: Properties panel ~260px */}
        <div className="w-[280px] flex-shrink-0 overflow-hidden flex flex-col">
          <RightPanel activeSegment={activeSegment} />
        </div>
      </div>

      {/* Bottom: Timeline */}
      <TimelineBar isPlaying={isPlaying} onToggle={() => setIsPlaying(p => !p)} />
    </div>
  )
}
