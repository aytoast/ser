import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Allow Turbopack (Next.js 16 default) — WASM files served from public/
  turbopack: {},
};

export default nextConfig;
