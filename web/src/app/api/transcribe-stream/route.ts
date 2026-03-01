/**
 * SSE proxy: forwards audio to Modal /transcribe/stream and pipes tokens back.
 * Browser cannot call Modal directly (no CORS), so this Next.js route acts as relay.
 */

const MODAL_API_URL =
  process.env.MODAL_API_URL ??
  "https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run";

export async function POST(req: Request) {
  const formData = await req.formData();
  const audioFile = formData.get("audio") as Blob | null;

  if (!audioFile) {
    return new Response(JSON.stringify({ error: "Audio file is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
  if (audioFile.size > MAX_UPLOAD_BYTES) {
    return new Response(
      JSON.stringify({ error: `File exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit` }),
      { status: 413, headers: { "Content-Type": "application/json" } }
    );
  }

  // Forward to Modal streaming endpoint, preserving original filename for format detection
  const upstream = new FormData();
  const originalName = (audioFile as File).name || "audio.wav";
  upstream.append("file", audioFile, originalName);

  const language = formData.get("language") as string | null;
  if (language) upstream.append("language", language);

  try {
    const res = await fetch(`${MODAL_API_URL}/transcribe/stream`, {
      method: "POST",
      body: upstream,
      signal: AbortSignal.timeout(5 * 60 * 1000),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "Upstream error");
      return new Response(
        JSON.stringify({ error: errText }),
        { status: res.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Pipe the SSE stream through to the client
    return new Response(res.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error: unknown) {
    const isTimeout =
      error instanceof Error &&
      (error.name === "TimeoutError" || error.name === "AbortError");
    return new Response(
      JSON.stringify({
        error: isTimeout ? "Transcription timed out" : "Failed to reach Modal API",
      }),
      { status: isTimeout ? 504 : 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
