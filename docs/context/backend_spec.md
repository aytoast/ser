## overview
- **model:** Ethostral — fine-tuned Mistral Voxtral for joint ASR and emotion classification.
- **framework:** Python (FastAPI) for the API layer.
- **inference runtime:** Hugging Face `transformers` + `peft` for adapter-based fine-tuned inference.
- **real-time transport:** WebSockets for streaming audio and transcription events.
- **hosting:** Hugging Face Inference Endpoints for model serving.

## api endpoints

### `POST /transcribe`
- **purpose:** accepts an uploaded audio/video file, runs the Ethostral pipeline, returns a structured transcript.
- **input:** multipart form-data with fields: `file`, `language` (optional, default: auto-detect), `diarize` (bool), `emotion` (bool).
- **output:** JSON object with diarized segments, each containing transcript text, speaker id, timestamps, and emotional metadata.

### `WS /transcribe/stream`
- **purpose:** accepts a live audio byte stream, emits partial transcription and emotion events in real time.
- **message format (server → client):**
  ```json
  {
    "segment_id": "uuid",
    "speaker": "s0",
    "text": "Hello, I'm here.",
    "start_ms": 8100,
    "end_ms": 9040,
    "emotion": {
      "label": "Calm",
      "valence": 0.3,
      "arousal": -0.1,
      "dominance": 0.2
    }
  }
  ```

### `GET /sessions/{session_id}`
- **purpose:** retrieves a previously processed session by ID.
- **output:** full structured transcript with emotional metadata.

### `DELETE /sessions/{session_id}`
- **purpose:** deletes a stored session.

## processing pipeline

1. **ingest:** audio is received via REST upload or WebSocket stream.
2. **preprocessing:** audio is resampled to 16 kHz mono. silence segments are stripped via VAD (voice activity detection).
3. **diarization:** speaker diarization using `pyannote.audio` to split audio into per-speaker segments.
4. **inference:** each segment is passed to the Ethostral endpoint for:
   - automatic speech recognition (ASR).
   - emotion classification (categorical + dimensional: valence / arousal / dominance).
5. **post-processing:** results are merged, timestamps are aligned, and output is structured per-segment.
6. **storage:** sessions are persisted with a generated UUID.
7. **telemetry:** each pipeline run is traced via Weights & Biases Weave.

## output schema

```typescript
type Segment = {
  id: string
  speaker: string           // "s0", "s1", ...
  start_ms: number
  end_ms: number
  text: string
  emotion: {
    label: string           // "Happy", "Neutral", "Anxious", etc.
    valence: number         // -1.0 to 1.0
    arousal: number         // -1.0 to 1.0
    dominance: number       // -1.0 to 1.0
    confidence: number      // 0.0 to 1.0
  }
}

type Session = {
  id: string
  filename: string
  language: string
  duration_ms: number
  created_at: string        // ISO 8601
  segments: Segment[]
}
```

## dependencies
- **`fastapi`** — async HTTP and WebSocket server.
- **`pydantic`** — request/response schema validation.
- **`pyannote.audio`** — speaker diarization.
- **`transformers` + `peft`** — Ethostral model loading and adapter inference.
- **`torchaudio`** — audio preprocessing and resampling.
- **`wandb`** — Weights & Biases Weave integration for pipeline tracing.
- **`huggingface_hub`** — programmatic access to model weights and datasets.

## performance targets
- **transcription latency (batch):** < 2× real-time (e.g., a 60s file processed in < 120s).
- **streaming latency:** < 500ms from audio chunk to partial transcript event.
- **emotion classification latency:** < 100ms per segment (excluding ASR).
- **word error rate:** target < 10% on clean English audio.
- **emotion F1 score:** target > 0.70 across the IEMOCAP benchmark.
