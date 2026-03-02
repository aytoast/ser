/**
 * Warmup endpoint: pings the Modal API /health to wake the GPU container.
 * Modal containers have a 5-min scaledown window, so hitting /health
 * during page load ensures the container is warm when recording starts.
 */

const MODAL_API_URL =
  process.env.MODAL_API_URL ??
  "https://yongkang-zou1999--evoxtral-api-evoxtralmodel-web.modal.run";

export async function GET() {
  try {
    const res = await fetch(`${MODAL_API_URL}/health`, {
      signal: AbortSignal.timeout(60_000), // Modal cold start can take up to 60s
    });
    const data = await res.json();
    return Response.json(data);
  } catch {
    return Response.json({ status: "unreachable" }, { status: 502 });
  }
}
