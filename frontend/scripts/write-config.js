#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const url = String(
  process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_URL || "",
)
  .trim()
  .replace(/\/$/, "");

if (process.env.VERCEL && !url) {
  console.warn(
    "[frontend] API_BASE_URL is not set. The Vercel UI will call same-origin paths and will not reach the GPU backend.",
  );
}

const dest = path.join(__dirname, "..", "src", "config.js");
const contents = `window.RETAIL_VISION_API_URL = ${JSON.stringify(url)};\n`;
fs.writeFileSync(dest, contents);
console.log(`[frontend] wrote ${path.relative(process.cwd(), dest)} API_BASE_URL=${url || "(same origin)"}`);
