"use client"

import React, { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Waves,
  Microphone,
  Export,
  Play,
  Pause,
  SpeakerHigh,
  Waveform,
  ChartLine,
  ChartPieSlice,
  ArrowsInLineVertical,
  Sliders
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

// --- Mock Data ---

const MOCK_TRANSCRIPT = [
  { id: 1, speaker: "Speaker 0", time: "00:10", text: "I can't believe we're actually doing this.", emotion: "Surprise", valence: 0.6, arousal: 0.8 },
  { id: 2, speaker: "Speaker 1", time: "00:15", text: "It's been a long time coming. Are you nervous?", emotion: "Neutral", valence: 0.1, arousal: 0.2 },
  { id: 3, speaker: "Speaker 0", time: "00:22", text: "A little bit. The data is just... it's a lot to process.", emotion: "Anxiety", valence: -0.4, arousal: 0.7 },
  { id: 4, speaker: "Speaker 1", time: "00:30", text: "Don't worry, the models are calibrated for this level of noise.", emotion: "Calm", valence: 0.3, arousal: -0.2 },
]

// --- Components ---

const Header = () => (
  <header className="h-14 border-b border-white/10 bg-black/40 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-50">
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse" />
        <span className="text-sm font-medium text-white/80">LIVE_SESSION_01.wav</span>
      </div>
      <div className="h-4 w-[1px] bg-white/10" />
      <span className="text-xs font-sans text-white/40">DURATION: 00:32:45</span>
    </div>
    <div className="flex items-center gap-3">
      <Button variant="outline" size="sm" className="bg-white/5 border-white/10 hover:bg-white/10 text-white/80 gap-2">
        <Export size={16} weight="bold" />
        EXPORT
      </Button>
      <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
        <div className="w-4 h-4 rounded-full bg-blue-500" />
      </div>
    </div>
  </header>
)

const TranscriptPanel = ({ activeSegment, onSelect }: any) => (
  <div className="flex flex-col h-full bg-black/20 border-r border-white/5">
    <div className="p-4 border-b border-white/5 flex items-center gap-2">
      <SpeakerHigh size={18} className="text-white/60" />
      <h2 className="text-xs font-bold tracking-widest text-white/60 uppercase">DIARIZED_TRANSCRIPT</h2>
    </div>
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-6">
        {MOCK_TRANSCRIPT.map((item) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className={`group cursor-pointer p-3 rounded-xl transition-all ${activeSegment === item.id ? 'bg-white/10 ring-1 ring-white/20' : 'hover:bg-white/5'}`}
            onClick={() => onSelect(item)}
          >
            <div className="flex items-start gap-3">
              <Avatar className="w-8 h-8 rounded-lg">
                <AvatarFallback className="bg-white/5 text-[10px] text-white/40">{item.speaker[item.speaker.length - 1]}</AvatarFallback>
              </Avatar>
              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold text-white/40 uppercase tracking-tighter">{item.speaker}</span>
                  <span className="text-[10px] font-sans text-white/20">{item.time}</span>
                </div>
                <p className="text-sm text-white/80 leading-relaxed font-sans">{item.text}</p>
                <div className="pt-2 flex flex-wrap gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Badge variant="outline" className="bg-blue-500/10 border-blue-500/30 text-blue-400 text-[9px] px-1.5 py-0 uppercase font-sans">
                          {item.emotion}
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent className="bg-black/90 border-white/10 text-xs">
                        Dominant Emotion Tag
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </ScrollArea>
  </div>
)

const QuadViewAnalytics = ({ activeSegment }: any) => {
  return (
    <div className="grid grid-cols-2 grid-rows-2 h-full gap-[1px] bg-white/5">
      {/* 1. Russell's Circumplex */}
      <Card className="rounded-none border-none bg-black/40 p-4 flex flex-col gap-4 relative overflow-hidden group">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ChartLine size={18} className="text-blue-400" />
            <h3 className="text-[10px] font-bold tracking-widest text-white/40 uppercase">RUSSELL_CIRCUMPLEX</h3>
          </div>
          <span className="text-[10px] font-sans text-white/20">V: {activeSegment?.valence || 0} / A: {activeSegment?.arousal || 0}</span>
        </div>
        <div className="flex-1 bg-white/5 rounded-lg relative flex items-center justify-center border border-white/5">
          <div className="absolute inset-0 flex items-center justify-center opacity-10">
            <div className="w-full h-[1px] bg-white" />
            <div className="h-full w-[1px] bg-white absolute" />
          </div>
          <div className="absolute top-2 left-1/2 -translate-x-1/2 text-[8px] text-white/20">AROUSAL +</div>
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[8px] text-white/20">AROUSAL -</div>
          <div className="absolute left-2 top-1/2 -translate-y-1/2 text-[8px] text-white/20 rotate-90">VALENCE -</div>
          <div className="absolute right-2 top-1/2 -translate-y-1/2 text-[8px] text-white/20 -rotate-90">VALENCE +</div>

          <motion.div
            animate={{
              x: (activeSegment?.valence || 0) * 100,
              y: (activeSegment?.arousal || 0) * -100
            }}
            className="w-4 h-4 bg-blue-500 rounded-full shadow-[0_0_20px_rgba(59,130,246,0.5)] border-2 border-white relative z-10"
          >
            <div className="absolute inset-0 animate-ping bg-blue-500 rounded-full opacity-50" />
          </motion.div>
        </div>
      </Card>

      {/* 2. Plutchik's Wheel / Radar */}
      <Card className="rounded-none border-none bg-black/40 p-4 flex flex-col gap-4 group">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ChartPieSlice size={18} className="text-purple-400" />
            <h3 className="text-[10px] font-bold tracking-widest text-white/40 uppercase">PLUTCHIK_WHEEL</h3>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center relative">
          <div className="w-32 h-32 rounded-full border border-white/10 flex items-center justify-center">
            <div className="w-20 h-20 rounded-full border border-white/5 flex items-center justify-center" />
            <div className="absolute inset-0 flex items-center justify-center">
              {[0, 45, 90, 135, 180, 225, 270, 315].map(deg => (
                <div key={deg} className="absolute w-[1px] h-full bg-white/5" style={{ transform: `rotate(${deg}deg)` }} />
              ))}
            </div>
            <motion.div
              animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
              className="w-24 h-24 bg-purple-500/20 rounded-full border border-purple-500/40 backdrop-blur-sm"
              style={{ clipPath: 'polygon(50% 0%, 90% 20%, 100% 60%, 75% 100%, 25% 100%, 0% 60%, 10% 20%)' }}
            />
          </div>
        </div>
      </Card>

      {/* 3. Prosodic Meters */}
      <Card className="rounded-none border-none bg-black/40 p-4 flex flex-col gap-4 group">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ArrowsInLineVertical size={18} className="text-emerald-400" />
            <h3 className="text-[10px] font-bold tracking-widest text-white/40 uppercase">PROSODIC_ANALYSIS</h3>
          </div>
        </div>
        <div className="flex-1 flex items-end justify-between px-4 pb-2 gap-4">
          {[
            { label: "PITCH", value: 160, max: 400, unit: "Hz", color: "bg-emerald-500" },
            { label: "JITTER", value: 0.12, max: 1, unit: "%", color: "bg-blue-500" },
            { label: "RATE", value: 140, max: 300, unit: "wpm", color: "bg-amber-500" },
            { label: "SHIM", value: 0.4, max: 1, unit: "dB", color: "bg-purple-500" },
          ].map(meter => (
            <div key={meter.label} className="flex-1 flex flex-col items-center gap-2 h-full">
              <div className="flex-1 w-full bg-white/5 rounded-sm overflow-hidden relative flex flex-col justify-end">
                <div className="absolute inset-0 opacity-10 flex flex-col justify-between p-1">
                  {[...Array(10)].map((_, i) => <div key={i} className="h-[1px] w-full bg-white" />)}
                </div>
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: `${(meter.value / meter.max) * 100}%` }}
                  className={`w-full ${meter.color} shadow-[0_0_15px_rgba(255,255,255,0.1)]`}
                />
              </div>
              <span className="text-[8px] font-bold text-white/20 uppercase tracking-tighter">{meter.label}</span>
              <span className="text-[9px] font-sans text-white/40">{meter.value}{meter.unit}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* 4. PAD Space */}
      <Card className="rounded-none border-none bg-black/40 p-4 flex flex-col gap-4 group">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders size={18} className="text-amber-400" />
            <h3 className="text-[10px] font-bold tracking-widest text-white/40 uppercase">PAD_DIMENSIONS</h3>
          </div>
        </div>
        <div className="flex-1 space-y-4 px-2">
          {[
            { label: "Pleasure", value: 45, color: "bg-emerald-500" },
            { label: "Arousal", value: 78, color: "bg-blue-500" },
            { label: "Dominance", value: 62, color: "bg-purple-500" },
          ].map(slider => (
            <div key={slider.label} className="space-y-1">
              <div className="flex justify-between text-[8px] uppercase tracking-widest text-white/40">
                <span>{slider.label}</span>
                <span className="font-sans">{slider.value}%</span>
              </div>
              <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${slider.value}%` }}
                  className={`h-full ${slider.color}`}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

const EmotionalTimeline = () => (
  <footer className="h-32 border-t border-white/10 bg-black/60 backdrop-blur-xl flex flex-col">
    <div className="h-2 w-full bg-white/5 relative">
      <div className="absolute left-1/4 w-[1px] h-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,1)] z-10" />
      <div className="absolute left-1/4 right-0 h-full bg-blue-500/10" />
    </div>
    <div className="flex-1 flex px-6 py-4 gap-6 items-center">
      <div className="flex items-center gap-4">
        <Button size="icon" variant="ghost" className="rounded-full hover:bg-white/10 text-white">
          <Waves size={24} weight="bold" />
        </Button>
        <div className="flex items-center gap-1">
          <Button size="icon" variant="ghost" className="h-8 w-8 text-white/60">
            <Play size={18} weight="fill" />
          </Button>
          <span className="text-[10px] font-sans text-white/40">00:32:04 / 01:20:00</span>
        </div>
      </div>
      <div className="flex-1 h-12 relative flex items-center justify-between gap-[2px]">
        {[...Array(120)].map((_, i) => (
          <div
            key={i}
            className={`w-[2px] rounded-full transition-all duration-300 ${i % 10 === 0 ? 'h-8 bg-white/20' : 'h-4 bg-white/5'}`}
            style={{ height: `${20 + Math.random() * 60}%`, opacity: i > 30 ? 0.3 : 1 }}
          />
        ))}
        <div className="absolute inset-x-0 h-[1px] bg-white/5" />
      </div>
      <div className="flex items-center gap-4">
        <div className="flex flex-col items-end">
          <span className="text-[8px] font-bold text-white/20 uppercase tracking-widest">SCANNING_RESOLUTION</span>
          <span className="text-[10px] font-sans text-white/60">120ms/sample</span>
        </div>
        <Slider defaultValue={[25]} max={100} step={1} className="w-24" />
      </div>
    </div>
  </footer>
)

const RecorderOverlay = () => {
  const [isRecording, setIsRecording] = useState(false)

  return (
    <div className="absolute right-8 bottom-40 z-50">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsRecording(!isRecording)}
        className={`w-16 h-16 rounded-full flex items-center justify-center relative transition-all duration-500 ${isRecording ? 'bg-red-500 shadow-[0_0_50px_rgba(239,68,68,0.5)]' : 'bg-white/10 hover:bg-white/20 backdrop-blur-md'}`}
      >
        <AnimatePresence mode="wait">
          {isRecording ? (
            <motion.div
              key="stop"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
            >
              <div className="w-6 h-6 bg-white rounded-sm" />
            </motion.div>
          ) : (
            <motion.div
              key="mic"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
            >
              <Microphone size={32} weight="bold" className="text-white" />
            </motion.div>
          )}
        </AnimatePresence>

        {isRecording && (
          <motion.div
            initial={{ scale: 1 }}
            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute inset-0 rounded-full border-2 border-red-500"
          />
        )}
      </motion.button>
    </div>
  )
}

export default function StudioPage() {
  const [activeSegment, setActiveSegment] = useState(MOCK_TRANSCRIPT[0])

  return (
    <main className="flex flex-col h-screen bg-[#050505] text-white selection:bg-blue-500/30 overflow-hidden font-sans">
      <Header />

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Transcript */}
        <div className="w-[400px]">
          <TranscriptPanel
            activeSegment={activeSegment.id}
            onSelect={setActiveSegment}
          />
        </div>

        {/* Right Panel: Analytics */}
        <div className="flex-1 flex flex-col relative">
          <QuadViewAnalytics activeSegment={activeSegment} />

          <div className="absolute inset-0 pointer-events-none border-[1px] border-white/5 z-20" />

          {/* Recorder Hub */}
          <RecorderOverlay />
        </div>
      </div>

      <EmotionalTimeline />
    </main>
  )
}
