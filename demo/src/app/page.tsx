"use client"

import React, { useState } from "react"
import Link from "next/link"
import { motion, AnimatePresence } from "framer-motion"
import {
  Microphone, MagnifyingGlass, DotsThreeVertical,
  UploadSimple, Sparkle, Play, Clock,
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

// --- Mock Data ---
const MOCK_SESSIONS = [
  { id: "1", title: "Team_Standup_2026-02-28.mp4", createdAt: "2 days ago" },
  { id: "2", title: "Customer_Interview_Batch_7.wav", createdAt: "5 days ago" },
  { id: "3", title: "Podcast_Episode_14.mp3", createdAt: "1 week ago" },
  { id: "4", title: "WeChat_20250804025710.mp4", createdAt: "7 months ago" },
]

// --- Upload Dialog ---
function UploadDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Transcribe files</DialogTitle>
        </DialogHeader>

        {/* Drop Zone */}
        <div className="border-2 border-dashed border-border rounded-lg px-6 py-10 flex flex-col items-center gap-2 text-center cursor-pointer hover:border-foreground/30 hover:bg-muted/40 transition-colors">
          <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center mb-1">
            <UploadSimple size={20} className="text-muted-foreground" />
          </div>
          <p className="text-sm font-semibold text-foreground">Click or drag files here to upload</p>
          <p className="text-xs text-muted-foreground">Audio & video files, up to 1000MB</p>
        </div>

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
                <SelectItem value="es">Spanish</SelectItem>
                <SelectItem value="fr">French</SelectItem>
                <SelectItem value="zh">Chinese</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="diarize" className="text-sm font-medium">Speaker diarization</Label>
            <Switch id="diarize" defaultChecked />
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

        <Button className="w-full gap-2 font-bold">
          <UploadSimple size={16} weight="bold" />
          Upload files
        </Button>
      </DialogContent>
    </Dialog>
  )
}

// --- Main Page ---
export default function HomePage() {
  const [showModal, setShowModal] = useState(false)
  const [search, setSearch] = useState("")

  const filtered = MOCK_SESSIONS.filter(s =>
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
              <span className="underline underline-offset-2 text-foreground font-medium cursor-pointer">industry-leading ASR model.</span>
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
          <MagnifyingGlass size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search transcripts..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9 h-9 text-sm bg-card"
          />
        </div>

        {/* Table */}
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-[11px] font-black uppercase tracking-widest text-muted-foreground">Title</TableHead>
              <TableHead className="text-[11px] font-black uppercase tracking-widest text-muted-foreground">Created at</TableHead>
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
              filtered.map((session, i) => (
                <TableRow key={session.id} className="cursor-pointer group">
                  <TableCell>
                    <Link href="/studio" className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-md bg-muted border border-border flex items-center justify-center flex-shrink-0">
                        <Play size={12} weight="fill" className="text-muted-foreground" />
                      </div>
                      <span className="text-sm font-semibold truncate max-w-xs">{session.title}</span>
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
                        <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity">
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
