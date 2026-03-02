"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import Link from "next/link"
import { ArrowLeft, VideoCamera, CircleNotch } from "@phosphor-icons/react"
import { useFER, FER_LABELS, type EmotionScores } from "@/hooks/use-fer"
import { useStreamingTranscription } from "@/hooks/use-streaming-transcription"

// ── Emotion bars overlay (bottom-left, transparent) ─────────────────────
function EmotionBars({ scores }: { scores: EmotionScores }) {
  return (
    <div className="absolute bottom-3 left-3 w-56 rounded-xl bg-black/40 backdrop-blur-md px-4 py-3">
      <div className="space-y-1.5">
        {FER_LABELS.map((label) => {
          const value = scores[label] ?? 0
          return (
            <div key={label} className="flex items-center gap-2">
              <div className="flex-1 h-[5px] bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-white/70 transition-all duration-300 ease-out"
                  style={{ width: `${Math.max(value * 100, 0.5)}%` }}
                />
              </div>
              <span className="text-[10px] text-white/70 font-medium w-14 text-right">
                {label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Warmup steps ────────────────────────────────────────────────────────
type WarmupStep = { label: string; status: "pending" | "loading" | "done" | "error" }

// ── Main page ───────────────────────────────────────────────────────────
export default function LivePage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const transcriptRef = useRef<HTMLDivElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const ferIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [cameraReady, setCameraReady] = useState(false)
  const [permissionError, setPermissionError] = useState<string | null>(null)
  const [warmupDone, setWarmupDone] = useState(false)
  const [warmupSteps, setWarmupSteps] = useState<WarmupStep[]>([
    { label: "Camera & Microphone", status: "pending" },
    { label: "FER Model (emotion detection)", status: "pending" },
    { label: "Evoxtral API (transcription)", status: "pending" },
  ])

  const { isLoaded: ferLoaded, scores, preload: preloadFER, classify } = useFER()
  const { isRecording, isTranscribing, transcript, currentChunk, start, stop, reset } =
    useStreamingTranscription()

  // ── Warmup sequence ─────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false

    const updateStep = (idx: number, status: WarmupStep["status"]) => {
      setWarmupSteps((prev) => prev.map((s, i) => (i === idx ? { ...s, status } : s)))
    }

    async function warmup() {
      // Step 1: Camera & Mic
      updateStep(0, "loading")
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setCameraReady(true)
        updateStep(0, "done")
      } catch (err: unknown) {
        if (cancelled) return
        updateStep(0, "error")
        if (err instanceof DOMException && err.name === "NotAllowedError") {
          setPermissionError(
            "Camera and microphone access was denied. Please allow permissions and reload."
          )
        } else {
          setPermissionError(
            "Could not access camera or microphone. Check your device settings."
          )
        }
        return
      }

      // Step 2: FER model (parallel with step 3)
      updateStep(1, "loading")
      const ferPromise = preloadFER()
        .then(() => { if (!cancelled) updateStep(1, "done") })
        .catch(() => { if (!cancelled) updateStep(1, "error") })

      // Step 3: Evoxtral API warmup — actually wake the Modal GPU container
      updateStep(2, "loading")
      const apiPromise = fetch("/api/warmup")
        .then((r) => r.json())
        .then((data) => {
          if (!cancelled) updateStep(2, data.status === "ok" ? "done" : "error")
        })
        .catch(() => { if (!cancelled) updateStep(2, "error") })

      await Promise.all([ferPromise, apiPromise])
      if (!cancelled) setWarmupDone(true)
    }

    warmup()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [preloadFER])

  // ── FER classification interval (always on when camera ready) ────────
  useEffect(() => {
    if (!ferLoaded || !cameraReady) return

    ferIntervalRef.current = setInterval(() => {
      if (videoRef.current) {
        classify(videoRef.current)
      }
    }, 500)

    return () => {
      if (ferIntervalRef.current) clearInterval(ferIntervalRef.current)
    }
  }, [ferLoaded, cameraReady, classify])

  // ── Auto-scroll transcript ────────────────────────────────────────────
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight
    }
  }, [transcript, currentChunk])

  // ── Handlers ──────────────────────────────────────────────────────────
  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      stop()
    } else {
      // Reuse the existing camera+mic stream so we don't request permissions again
      start(streamRef.current ?? undefined)
    }
  }, [isRecording, start, stop])

  const hasScores = Object.keys(scores).length > 0

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border/40">
        <Link
          href="/"
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft size={16} weight="bold" />
          <span>Back to Studio</span>
        </Link>

        <span className="text-sm font-semibold tracking-widest uppercase text-muted-foreground">
          Live Mode
        </span>

        <div className="flex items-center gap-2">
          {isRecording && (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500" />
            </span>
          )}
        </div>
      </header>

      {/* Body */}
      <div className="flex-1 flex flex-col items-center gap-6 px-4 py-8 max-w-4xl mx-auto w-full">
        {/* Camera preview */}
        <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-slate-950 border border-border/30 shadow-2xl">
          {permissionError ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-8 text-center">
              <VideoCamera size={48} className="text-muted-foreground/40" weight="thin" />
              <p className="text-sm text-muted-foreground max-w-md">{permissionError}</p>
            </div>
          ) : !cameraReady ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <VideoCamera size={48} className="text-muted-foreground/30 animate-pulse" weight="thin" />
              <p className="text-xs text-muted-foreground/50">Initializing camera...</p>
            </div>
          ) : null}

          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />

          {/* Emotion bars overlay */}
          {cameraReady && hasScores && <EmotionBars scores={scores} />}

          {/* Record button — centred at bottom of video */}
          {cameraReady && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3">
              <button
                onClick={handleToggleRecording}
                disabled={!warmupDone || isTranscribing}
                className="group relative flex items-center justify-center"
                aria-label={isRecording ? "Stop recording" : "Start recording"}
              >
                {/* outer ring */}
                <span
                  className={`absolute size-16 rounded-full border-[3px] transition-colors ${
                    isRecording ? "border-white" : isTranscribing ? "border-white/30" : "border-white/60"
                  }`}
                />
                {/* inner shape */}
                <span
                  className={`transition-all duration-200 ${
                    !warmupDone || isTranscribing
                      ? "size-10 rounded-full bg-white/30 animate-pulse"
                      : isRecording
                        ? "size-6 rounded-[4px] bg-red-500"
                        : "size-12 rounded-full bg-red-500 group-hover:bg-red-400"
                  }`}
                />
              </button>

              {/* Reset button */}
              {(transcript || currentChunk) && !isRecording && !isTranscribing && (
                <button
                  onClick={reset}
                  className="text-[11px] text-white/60 hover:text-white/90 uppercase tracking-wider transition-colors"
                >
                  Reset
                </button>
              )}
            </div>
          )}

          {/* Transcript overlay — bottom-right, glassmorphism */}
          {cameraReady && (transcript || currentChunk || isTranscribing) && (
            <div
              ref={transcriptRef}
              className="absolute bottom-3 right-3 w-64 max-h-32 overflow-y-auto rounded-xl bg-black/40 backdrop-blur-md px-4 py-3 text-[12px] leading-relaxed text-white/90"
            >
              {transcript}
              {currentChunk && (
                <span className="text-white/50">{currentChunk}</span>
              )}
              {isTranscribing && !currentChunk && (
                <span className="text-white/40 italic">Transcribing...</span>
              )}
            </div>
          )}
        </div>

        {/* Warmup progress (shown before ready) */}
        {!warmupDone && (
          <div className="w-full rounded-xl bg-muted/30 border border-border/30 p-5 space-y-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/60">
              Warming up...
            </p>
            {warmupSteps.map((step, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                {step.status === "pending" && (
                  <div className="size-4 rounded-full border-2 border-muted-foreground/20" />
                )}
                {step.status === "loading" && (
                  <CircleNotch size={16} className="text-blue-500 animate-spin" weight="bold" />
                )}
                {step.status === "done" && (
                  <div className="size-4 rounded-full bg-emerald-500 flex items-center justify-center">
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                      <path d="M2 5L4 7L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                )}
                {step.status === "error" && (
                  <div className="size-4 rounded-full bg-red-500 flex items-center justify-center">
                    <span className="text-[10px] text-white font-bold">!</span>
                  </div>
                )}
                <span className={step.status === "done" ? "text-foreground/70" : step.status === "loading" ? "text-foreground" : "text-muted-foreground/50"}>
                  {step.label}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
