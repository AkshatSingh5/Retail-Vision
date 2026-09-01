#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const url = String(
  process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_URL || "",
)
  .trim()
  .replace(/\/$/, "");

function hostnameOf(value) {
  try {
    return new URL(value).hostname.toLowerCase();
  } catch (_e) {
    return "";
  }
}

function isVercelHost(host) {
  return host === "vercel.app" || host.endsWith(".vercel.app");
}

if (process.env.VERCEL) {
  if (!url) {
    console.error(
      "[frontend] API_BASE_URL is not set.\n" +
        "Vercel hosts the POS UI only. /cart, /products, and /pos/status live on the GPU FastAPI backend.\n" +
        "Set API_BASE_URL=https://YOUR-BACKEND-DOMAIN (no trailing slash) in Vercel env vars and redeploy.\n" +
        "Do not set it to this Vercel URL.",
    );
    process.exit(1);
  }
  const host = hostnameOf(url);
  if (!host) {
    console.error(`[frontend] API_BASE_URL is not a valid absolute URL: ${url}`);
    process.exit(1);
  }
  if (isVercelHost(host)) {
    console.error(
      "[frontend] API_BASE_URL points at Vercel. That host is the static UI, not FastAPI.\n" +
        "Set API_BASE_URL to the public HTTPS origin of the GPU backend.",
    );
    process.exit(1);
  }
}

const dest = path.join(__dirname, "..", "src", "config.js");
const contents = `window.RETAIL_VISION_API_URL = ${JSON.stringify(url)};\n`;
fs.writeFileSync(dest, contents);
console.log(`[frontend] wrote ${path.relative(process.cwd(), dest)} API_BASE_URL=${url || "(same origin)"}`);
