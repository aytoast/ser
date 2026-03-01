# emotional speech recognition and synthesis - product specification

## objective
build a pipeline that ingests human speech, classifies emotion across multiple frameworks, and transcribes text.

## inputs and outputs

### 1. input: audio injection
- **source:** raw speech material (16khz, mono) recorded via microphone.
- **process:** audio is streamed to the fine-tuned mistral voxtral endpoint for processing.

### 2. output: emotional transcription
- **content:** transcript interleaved with tags from multiple emotional frameworks.
- **supported frameworks & reference datasets:**
  - **russell’s circumplex model:** valence and arousal coordinates (Ref: IEMOCAP, EMOVOME).
  - **pad emotion space:** pleasure, arousal, and dominance dimensions (Ref: IEMOCAP).
  - **plutchik’s wheel:** categorical tags (Ref: RAVDESS, EmoDB).
  - **prosodic analysis:** pitch, jitter, and speech rate metadata (Ref: RAVDESS, Berlin Emo-db).
  - **stress & deception markers:** specialized detection (Ref: SUSAS, MDPE, CRISIS).
- **example output:** `[arousal: high, valence: negative] [stress: probable] [pitch: 210hz] i can't figure this out!`

## input/output payload definitions

### 1. speech-to-text request
- **endpoint:** `/v1/recognize`
- **payload:** binary audio/wav
- **constraints:** max duration 30 seconds.

### 2. speech-to-text response
- **format:** json
- **schema:**
  ```json
  {
    "text": "i can't figure this out.",
    "emotion": "frustrated",
    "confidence": 0.89
  }
  ```

## error handling
- **unrecognized emotion:** default to "neutral" tag.
- **api timeouts:** retry once, then fallback to standard text output.
- **audio format errors:** reject request with 400 bad request and specify required formatting (16khz, mono).

## resources
- [front end specification](./frontend_spec.md)