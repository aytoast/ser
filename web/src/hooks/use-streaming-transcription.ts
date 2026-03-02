"use client"

import { useState, useRef, useCallback, useEffect } from "react"

interface ChunkStreamEvent {
  token?: string
  done?: boolean
  transcription?: string
}

/**
 * Press-to-record transcription hook.
 * 1. start() → begins recording audio
 * 2. stop()  → stops recording, sends full audio to API, streams back tokens
 */
export function useStreamingTranscription() {
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [currentChunk, setCurrentChunk] = useState("")

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const ownStreamRef = useRef<MediaStream | null>(null)
  const blobsRef = useRef<Blob[]>([])

  const transcribe = useCallback(async (audioBlob: Blob) => {
    if (audioBlob.size < 500) return

    setIsTranscribing(true)
    setCurrentChunk("")

    try {
      const formData = new FormData()
      formData.append("audio", audioBlob, "recording.webm")

      const response = await fetch("/api/transcribe-stream", {
        method: "POST",
        body: formData,
      })

      if (!response.ok || !response.body) {
        console.error("Transcription request failed:", response.status)
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamingText = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith("data: ")) continue
          const jsonStr = trimmed.slice(6)
          if (!jsonStr) continue

          try {
            const event: ChunkStreamEvent = JSON.parse(jsonStr)
            if (event.done && event.transcription != null) {
              // Final result — set as transcript
              setTranscript((prev) =>
                prev ? prev + " " + event.transcription! : event.transcription!
              )
              setCurrentChunk("")
            } else if (event.token != null) {
              streamingText += event.token
              setCurrentChunk(streamingText)
            }
          } catch {
            // ignore malformed JSON
          }
        }
      }
    } catch (error) {
      console.error("Transcription error:", error)
    } finally {
      setIsTranscribing(false)
    }
  }, [])

  const start = useCallback(
    async (existingStream?: MediaStream) => {
      if (isRecording) return

      blobsRef.current = []

      let stream: MediaStream
      if (existingStream) {
        const audioTracks = existingStream.getAudioTracks()
        if (audioTracks.length === 0) {
          console.error("[useStreamingTranscription] No audio tracks")
          return
        }
        stream = new MediaStream(audioTracks)
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        ownStreamRef.current = stream
      }

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm"

      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          blobsRef.current.push(event.data)
        }
      }

      recorder.start()
      setIsRecording(true)
    },
    [isRecording]
  )

  const stop = useCallback(() => {
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state === "inactive") return

    // When the recorder stops, assemble blobs and send for transcription
    recorder.onstop = () => {
      const mimeType = recorder.mimeType || "audio/webm;codecs=opus"
      const audioBlob = new Blob(blobsRef.current, { type: mimeType })
      blobsRef.current = []
      transcribe(audioBlob)
    }

    recorder.stop()
    mediaRecorderRef.current = null

    // Only stop tracks if we own the stream
    if (ownStreamRef.current) {
      ownStreamRef.current.getTracks().forEach((t) => t.stop())
      ownStreamRef.current = null
    }

    setIsRecording(false)
  }, [transcribe])

  const reset = useCallback(() => {
    setTranscript("")
    setCurrentChunk("")
  }, [])

  useEffect(() => {
    return () => {
      const recorder = mediaRecorderRef.current
      if (recorder && recorder.state !== "inactive") recorder.stop()
      if (ownStreamRef.current) {
        ownStreamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  return {
    isRecording,
    isTranscribing,
    transcript,
    currentChunk,
    start,
    stop,
    reset,
  }
}
