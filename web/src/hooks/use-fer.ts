"use client"

import { useState, useRef, useCallback, useEffect } from "react"

export const FER_LABELS = [
  "Anger",
  "Contempt",
  "Disgust",
  "Fear",
  "Happy",
  "Neutral",
  "Sad",
  "Surprise",
] as const

export type EmotionScores = Record<string, number>

const IMAGE_SIZE = 224

const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]

function softmax(scores: Float32Array): Float32Array {
  const max = Math.max(...scores)
  const exps = new Float32Array(scores.length)
  let sum = 0
  for (let i = 0; i < scores.length; i++) {
    exps[i] = Math.exp(scores[i] - max)
    sum += exps[i]
  }
  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum
  }
  return exps
}

export function useFER(): {
  isLoaded: boolean
  emotion: string | null
  confidence: number
  scores: EmotionScores
  preload: () => Promise<void>
  classify: (video: HTMLVideoElement) => Promise<void>
} {
  const [isLoaded, setIsLoaded] = useState(false)
  const [emotion, setEmotion] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [scores, setScores] = useState<EmotionScores>({})

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sessionRef = useRef<any>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const loadingRef = useRef(false)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ortRef = useRef<any>(null)

  const loadModel = useCallback(async () => {
    if (sessionRef.current || loadingRef.current) return
    loadingRef.current = true

    try {
      const ort = await import("onnxruntime-web")
      ortRef.current = ort

      // Configure WASM before creating session
      ort.env.wasm.numThreads = 1
      ort.env.wasm.wasmPaths = "/"

      // Fetch model as ArrayBuffer
      const response = await fetch("/emotion_model_web.onnx")
      const modelBuffer = await response.arrayBuffer()

      const session = await ort.InferenceSession.create(
        new Uint8Array(modelBuffer),
        { executionProviders: ["wasm"] }
      )
      sessionRef.current = session
      setIsLoaded(true)
    } catch (err) {
      console.error("[useFER] Failed to load ONNX model:", err)
      loadingRef.current = false
    }
  }, [])

  const classify = useCallback(
    async (video: HTMLVideoElement) => {
      try {
        if (!sessionRef.current) {
          await loadModel()
        }

        const session = sessionRef.current
        const ort = ortRef.current
        if (!session || !ort) return

        if (!canvasRef.current) {
          canvasRef.current = document.createElement("canvas")
          canvasRef.current.width = IMAGE_SIZE
          canvasRef.current.height = IMAGE_SIZE
        }

        const canvas = canvasRef.current
        const ctx = canvas.getContext("2d", { willReadFrequently: true })
        if (!ctx) return

        ctx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE)
        const { data } = imageData

        const floatData = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE)
        const pixelCount = IMAGE_SIZE * IMAGE_SIZE

        for (let i = 0; i < pixelCount; i++) {
          const srcIdx = i * 4
          floatData[i] = (data[srcIdx] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
          floatData[pixelCount + i] =
            (data[srcIdx + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
          floatData[2 * pixelCount + i] =
            (data[srcIdx + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        }

        const inputTensor = new ort.Tensor("float32", floatData, [
          1,
          3,
          IMAGE_SIZE,
          IMAGE_SIZE,
        ])

        const inputName = session.inputNames[0]
        const results = await session.run({ [inputName]: inputTensor })
        const outputName = session.outputNames[0]
        const output = results[outputName]

        if (!output) return

        const rawScores = output.data as Float32Array
        const probs = softmax(rawScores)

        const scoreMap: EmotionScores = {}
        let maxIdx = 0
        let maxVal = probs[0]
        for (let i = 0; i < probs.length; i++) {
          scoreMap[FER_LABELS[i]] = probs[i]
          if (probs[i] > maxVal) {
            maxVal = probs[i]
            maxIdx = i
          }
        }

        setScores(scoreMap)
        setEmotion(FER_LABELS[maxIdx])
        setConfidence(maxVal)
      } catch (err) {
        console.error("[useFER] Classification error:", err)
      }
    },
    [loadModel]
  )

  useEffect(() => {
    return () => {
      if (sessionRef.current) {
        sessionRef.current.release().catch((err: unknown) => {
          console.error("[useFER] Failed to release session:", err)
        })
        sessionRef.current = null
      }
    }
  }, [])

  return { isLoaded, emotion, confidence, scores, preload: loadModel, classify }
}
