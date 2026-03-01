import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const audioFile = formData.get("audio") as Blob;

    if (!audioFile) {
      return NextResponse.json(
        { error: "Audio file is required" },
        { status: 400 }
      );
    }

    const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100MB
    if (audioFile.size > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        { error: `File size exceeds ${MAX_UPLOAD_BYTES / 1024 / 1024}MB limit` },
        { status: 413 }
      );
    }

    const MODEL_URL = process.env.MODEL_URL || "http://127.0.0.1:8000";

    // Forward the formData to the Python backend
    const response = await fetch(`${MODEL_URL}/transcribe`, {
      method: "POST",
      body: formData,
      // Signal timeout after 5 minutes
      signal: AbortSignal.timeout(5 * 60 * 1000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.detail || "Transcription failed at Model layer" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("Transcription proxy error:", error);
    const isTimeout = error.name === "TimeoutError" || error.name === "AbortError";
    return NextResponse.json(
      { error: isTimeout ? "Transcription timed out" : "Internal server error connecting to model layer" },
      { status: isTimeout ? 504 : 500 }
    );
  }
}
