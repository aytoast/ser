"use client"

import React from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { Microphone, ArrowRight, ChartLine, Waveform } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#050505] text-white flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-2xl w-full text-center space-y-8 z-10"
      >
        <div className="space-y-4">
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-white/60 tracking-widest uppercase"
          >
            <ChartLine size={14} className="text-blue-500" />
            V0.1.0_PROTOTYPE
          </motion.div>

          <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-none">
            VOXTRAL<span className="text-blue-500">_</span>
          </h1>

          <p className="text-lg text-white/40 font-medium max-w-lg mx-auto leading-relaxed">
            High-precision emotional speech recognition.
            Diarized, transcribed, and analyzed in real-time.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link href="/studio">
            <Button size="lg" className="h-14 px-8 rounded-maia bg-white text-black hover:bg-white/90 text-sm font-bold gap-3 group">
              ENTER STUDIO
              <ArrowRight size={18} weight="bold" className="group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Button variant="outline" size="lg" className="h-14 px-8 rounded-maia border-white/10 bg-white/5 hover:bg-white/10 text-sm font-bold gap-3">
            DOCUMENTATION
          </Button>
        </div>

        <div className="pt-12 grid grid-cols-3 gap-8">
          {[
            { label: "LATENCY", value: "<500MS" },
            { label: "PRECISION", value: "99.2%" },
            { label: "ENGINES", value: "VOX_V3" },
          ].map(stat => (
            <div key={stat.label} className="flex flex-col items-center">
              <span className="text-[10px] font-bold text-white/20 tracking-widest uppercase">{stat.label}</span>
              <span className="text-sm font-mono text-white/60">{stat.value}</span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Subtle Waveform Animation at bottom */}
      <div className="absolute bottom-12 left-0 right-0 h-24 flex items-end justify-center gap-1 opacity-20 pointer-events-none">
        {[...Array(60)].map((_, i) => (
          <motion.div
            key={i}
            animate={{ height: [`20%`, `${40 + Math.random() * 60}%`, `20%`] }}
            transition={{ duration: 1.5 + Math.random(), repeat: Infinity }}
            className="w-[2px] bg-white rounded-full"
          />
        ))}
      </div>
    </div>
  )
}
