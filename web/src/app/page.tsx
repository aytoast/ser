"use client"

import React, { useState, useRef } from "react"
import { useRouter } from "next/navigation"
import {
  UploadSimple, Microphone, VideoCamera, Waveform, CheckCircle, X,
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Navbar } from "@/components/navbar"
import { createPendingSession, type DiarizeResult } from "@/lib/session-store"
import { cn } from "@/lib/utils"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000"
const MAX_FILE_BYTES = 100 * 1024 * 1024

export default function CreatePage() {
  const router = useRouter()
  const inputRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState("")
  const [error, setError] = useState<string | null>(null)

  const isValidFile = (f: File) =>
    f.type.startsWith("audio/") ||
    f.type.startsWith("video/") ||
    /\.(wav|mp3|m4a|webm|ogg|flac|mp4|mov)$/i.test(f.name)

  const setFileIfValid = (f: File) => {
    if (!isValidFile(f)) {
      setError("Unsupported file type. Please upload audio or video.")
      return
    }
    if (f.size > MAX_FILE_BYTES) {
      setError(`File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max 100 MB.`)
      return
    }
    setFile(f)
    setError(null)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) setFileIfValid(f)
  }

  const handleUpload = async () => {
    if (!file) return
    const id = createPendingSession(file)
    router.push(`/studio?s=${id}`)
  }

  const isAudio = file?.type.startsWith("audio/") || /\.(wav|mp3|m4a|webm|ogg|flac)$/i.test(file?.name ?? "")

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />

      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 gap-12">
        {/* Welcome */}
        <div className="text-center space-y-3 max-w-lg">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-muted text-xs font-medium text-muted-foreground mb-2">
            <Waveform size={12} weight="fill" className="text-primary" />
            Emotional Speech Recognition
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            Create a new session
          </h1>
          <p className="text-muted-foreground text-sm leading-relaxed">
            Upload an audio or video file and Ethos Studio will transcribe it,
            identify speakers, and analyse emotional tone — all powered by Voxtral.
          </p>
        </div>

        {/* Upload Zone */}
        <div className="w-full max-w-xl space-y-4">
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,video/*,.wav,.mp3,.m4a,.webm,.ogg,.flac,.mp4,.mov"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) setFileIfValid(f)
            }}
          />

          {!file ? (
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={cn(
                "w-full border-2 border-dashed rounded-lg p-12 flex flex-col items-center gap-4 transition-colors cursor-pointer",
                dragging
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-muted-foreground/40 hover:bg-muted/30"
              )}
            >
              <div className="size-12 rounded-full bg-muted flex items-center justify-center">
                <UploadSimple size={22} className="text-muted-foreground" />
              </div>
              <div className="text-center space-y-1">
                <p className="text-sm font-semibold">Drop your file here, or click to browse</p>
                <p className="text-xs text-muted-foreground">Audio & video files · MP3, WAV, MP4, MOV, M4A · up to 100 MB</p>
              </div>

              {/* Type hints */}
              <div className="flex items-center gap-3 mt-2">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Microphone size={13} weight="fill" />
                  Audio
                </div>
                <span className="text-border">·</span>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <VideoCamera size={13} weight="fill" />
                  Video
                </div>
              </div>
            </button>
          ) : (
            <div className="border border-border rounded-lg p-5 flex items-center gap-4 bg-muted/20">
              <div className="size-10 rounded-lg bg-muted flex items-center justify-center shrink-0">
                {isAudio
                  ? <Microphone size={18} weight="fill" className="text-blue-500" />
                  : <VideoCamera size={18} weight="fill" className="text-pink-500" />
                }
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(1)} MB · {isAudio ? "Audio" : "Video"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle size={16} weight="fill" className="text-green-500 shrink-0" />
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-muted-foreground"
                  onClick={() => { setFile(null); setError(null) }}
                >
                  <X size={14} />
                </Button>
              </div>
            </div>
          )}

          {error && (
            <p className="text-sm text-destructive font-medium text-center">{error}</p>
          )}

          <Button
            className="w-full gap-2 font-semibold h-10"
            disabled={!file || loading}
            onClick={handleUpload}
          >
            {loading ? (
              <>
                <div className="size-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                {progress || "Processing…"}
              </>
            ) : (
              <>
                <Waveform size={16} weight="bold" />
                Transcribe & Analyse
              </>
            )}
          </Button>
        </div>
      </main>
    </div>
  )
}
