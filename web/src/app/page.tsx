"use client"

import React, { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import {
  Microphone, MagnifyingGlass, DotsThreeVertical,
  UploadSimple, Play, Clock,
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Navbar } from "@/components/navbar"
import { Skeleton } from "@/components/ui/skeleton"
import { createSession, type DiarizeResult } from "@/lib/session-store"

// --- Constants ---
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000"
const MAX_FILE_BYTES = 100 * 1024 * 1024

// --- Mock sessions (demo history) ---
const MOCK_SESSIONS = [
  { id: "demo-1", title: "Team_Standup_2026-02-28.mp4", createdAt: "2 days ago" },
  { id: "demo-2", title: "Customer_Interview_Batch_7.wav", createdAt: "5 days ago" },
  { id: "demo-3", title: "Podcast_Episode_14.mp3", createdAt: "1 week ago" },
  { id: "demo-4", title: "WeChat_20250804025710.mp4", createdAt: "7 months ago" },
]

// --- Upload Dialog ---
function UploadDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
}) {
  const router = useRouter()
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const inputRef = React.useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null
    setFile(f)
    setError(null)
    setProgress("")
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (
      f &&
      (f.type.startsWith("audio/") ||
        f.type.startsWith("video/") ||
        /\.(wav|mp3|m4a|webm|ogg|flac|mp4)$/i.test(f.name))
    ) {
      setFile(f)
      setError(null)
      setProgress("")
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an audio or video file")
      return
    }
    if (file.size > MAX_FILE_BYTES) {
      setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 100 MB.`)
      return
    }

    setLoading(true)
    setError(null)
    setProgress("Uploading…")

    const isVideo = file.type.startsWith("video/") || /\.(mp4|mkv|avi|mov|m4v)$/i.test(file.name)

    try {
      const formData = new FormData()
      formData.append("audio", file, file.name)

      setProgress(isVideo
        ? "Transcribing and analyzing facial emotions…"
        : "Transcribing and analyzing speakers…"
      )
      const res = await fetch(
        `${API_BASE}/api/transcribe-diarize`,
        { method: "POST", body: formData }
      )
      const data = await res.json().catch(() => ({}))

      if (!res.ok) {
        setError((data as { error?: string }).error ?? "Transcription failed")
        return
      }

      setProgress("Processing results…")
      const result = data as DiarizeResult
      const sessionId = createSession(file, result)

      // Navigate to studio with session ID
      router.push(`/studio?s=${sessionId}`)
      onOpenChange(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed")
    } finally {
      setLoading(false)
      setProgress("")
    }
  }

  const handleOpenChange = (v: boolean) => {
    if (!v && !loading) {
      setFile(null)
      setError(null)
      setProgress("")
    }
    if (!loading) onOpenChange(v)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Transcribe files</DialogTitle>
        </DialogHeader>

        {/* Drop Zone */}
        <div
          className={`border-2 border-dashed rounded-lg px-6 py-10 flex flex-col items-center gap-2 text-center transition-colors ${
            loading
              ? "border-border opacity-50 cursor-not-allowed"
              : "border-border cursor-pointer hover:border-foreground/30 hover:bg-muted/40"
          }`}
          onClick={() => !loading && inputRef.current?.click()}
          onDrop={loading ? undefined : handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,video/*,.wav,.mp3,.m4a,.webm,.ogg,.flac,.mp4"
            className="hidden"
            onChange={handleFileChange}
            disabled={loading}
          />
          <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center mb-1">
            {loading ? (
              <div className="w-5 h-5 border-2 border-foreground/30 border-t-foreground rounded-full animate-spin" />
            ) : (
              <UploadSimple size={20} className="text-muted-foreground" />
            )}
          </div>
          <p className="text-sm font-semibold text-foreground">
            {loading ? progress : "Click or drag files here to upload"}
          </p>
          <p className="text-xs text-muted-foreground">
            {file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : "Audio & video files, up to 100 MB"}
          </p>
        </div>

        {error && (
          <p className="text-sm text-destructive font-medium">{error}</p>
        )}

        <Separator />

        {/* Settings */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="lang-select" className="text-sm font-medium">Primary language</Label>
            <Select defaultValue="detect">
              <SelectTrigger id="lang-select" className="w-32 h-8 text-sm">
                <SelectValue placeholder="Detect" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="detect">Detect</SelectItem>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="zh">Chinese</SelectItem>
                <SelectItem value="es">Spanish</SelectItem>
                <SelectItem value="fr">French</SelectItem>
              </SelectContent>
            </Select>
          </div>

<div className="flex items-center justify-between">
            <Label htmlFor="diarize" className="text-sm font-medium">Speaker diarization</Label>
            <Switch id="diarize" defaultChecked disabled />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="emotion" className="text-sm font-medium">Emotion analysis</Label>
            <Switch id="emotion" defaultChecked />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="subtitles" className="text-sm font-medium">Include subtitles</Label>
            <Switch id="subtitles" />
          </div>
        </div>

        <Button
          className="w-full gap-2 font-bold"
          onClick={handleUpload}
          disabled={loading || !file}
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-primary-foreground/40 border-t-primary-foreground rounded-full animate-spin" />
              {progress || "Processing…"}
            </>
          ) : (
            <>
              <UploadSimple size={16} weight="bold" />
              Upload & transcribe
            </>
          )}
        </Button>
      </DialogContent>
    </Dialog>
  )
}

// --- Main Page ---
export default function HomePage() {
  const [showModal, setShowModal] = useState(false)
  const [search, setSearch] = useState("")

  const filtered = MOCK_SESSIONS.filter((s) =>
    s.title.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <main className="max-w-4xl mx-auto px-6 py-10">
        {/* Page Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-black tracking-tight">Speech to text</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Transcribe audio and video files with our{" "}
              <span className="underline underline-offset-2 text-foreground font-medium cursor-pointer">
                industry-leading ASR model.
              </span>
            </p>
          </div>
          <Button onClick={() => setShowModal(true)} className="gap-2 font-bold shadow-sm">
            <Microphone size={16} weight="bold" />
            Transcribe files
          </Button>
        </div>

        {/* Promo Banner */}
        <div className="border border-border rounded-lg p-4 flex items-center gap-4 mb-5 bg-card hover:bg-muted/40 transition-colors cursor-pointer">
          <Skeleton className="w-14 h-14 rounded-lg bg-foreground/10 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold">Try Ethostral Realtime</p>
            <p className="text-sm text-muted-foreground mt-0.5 leading-snug">
              Experience lightning-fast transcription with unmatched emotional accuracy, powered by Ethostral.
            </p>
          </div>
          <Button variant="outline" size="sm" className="flex-shrink-0 font-bold">
            Try the demo
          </Button>
        </div>

        {/* Search */}
        <div className="relative mb-5">
          <MagnifyingGlass
            size={15}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            placeholder="Search transcripts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 h-9 text-sm bg-card"
          />
        </div>

        {/* Table */}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-[11px] font-black uppercase tracking-widest text-muted-foreground">
                Title
              </TableHead>
              <TableHead className="text-[11px] font-black uppercase tracking-widest text-muted-foreground">
                Created at
              </TableHead>
              <TableHead className="w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow>
                <TableCell colSpan={3} className="text-center text-muted-foreground py-16 text-sm">
                  No transcripts found.
                </TableCell>
              </TableRow>
            ) : (
              filtered.map((session) => (
                <TableRow key={session.id} className="cursor-pointer group">
                  <TableCell>
                    {/* Demo sessions link to studio with ?demo=1 (shows mock data) */}
                    <Link href="/studio?demo=1" className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-md bg-muted border border-border flex items-center justify-center flex-shrink-0">
                        <Play size={12} weight="fill" className="text-muted-foreground" />
                      </div>
                      <span className="text-sm font-semibold truncate max-w-xs">
                        {session.title}
                      </span>
                    </Link>
                  </TableCell>
                  <TableCell>
                    <span className="text-sm text-muted-foreground flex items-center gap-1.5">
                      <Clock size={13} />
                      {session.createdAt}
                    </span>
                  </TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <DotsThreeVertical size={16} />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem>Open</DropdownMenuItem>
                        <DropdownMenuItem>Export</DropdownMenuItem>
                        <DropdownMenuItem className="text-destructive">Delete</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </main>

      <UploadDialog open={showModal} onOpenChange={setShowModal} />
    </div>
  )
}
