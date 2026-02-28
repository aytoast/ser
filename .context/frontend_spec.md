## visual identity
- **typography:** high-precision monospaced fonts (JetBrains Mono) for data, alongside maia-style sans-serif for UI.
- **componentry:** shadcn ui base.
- **style profile:** maia (radix-maia) — characterized by soft rounding, high-contrast surfaces, and subtle glassmorphism.
- **iconography:** phosphor icons (thin or regular weight).

## layout & components
1. **header:** metadata bar showing filename, session duration, and a global "Export" button (JSON/CSV).
2. **diarized transcript (left panel):** structured list of utterances grouped by speaker. each bubble contains the text and an "emotion chip" that reveals the dominant tags on hover.
3. **analytics dashboard (right panel - "the quad-view"):**
   - **russell’s circumplex:** a 2D coordinate grid with a tracking point moving between valence and arousal axes.
   - **plutchik’s wheel:** a radar chart showing intensities across primary emotions.
   - **prosodic meters:** vertical "V-U" style meters for pitch (Hz), jitter (%), and speech rate (wpm).
   - **pad space:** 3D sliders representing pleasure, arousal, and dominance indices.
4. **emotional timeline (bottom panel):** horizontal waveform view with color-coded sentiment "heatmaps" (e.g., a crimson stretch for stress). includes playback controls and a zoom slider.
5. **recorder hub (center):** circular "record" button with a real-time reactive glow and oscilloscope-style waveform animation.

## interactions
- **diarization selection:** clicking a speaker avatar filters the analytics to only show their emotional profile.
- **segment scrubbing:** clicking a transcript segment syncs the timeline and the quad-view analytics to that specific moment.
- **real-time playback:** users can replay specific chunks of audio with an overlay showing the live-calculated emotion data.

## technical requirements
- **real-time synchronicity:** ensuring the transcription, timeline, and 4-way analytics update in perfect sync with sub-500ms latency.
- **framing:** Next.js with Framer Motion for premium transitions.
- **ui library:** shadcn/ui with maia theme configuration.
- **icons:** @phosphor-icons/react.
