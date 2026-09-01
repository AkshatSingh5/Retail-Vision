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

function isPlaceholder(value) {
  return /your-backend-domain|example\.com|changeme/i.test(value);
}

if (process.env.VERCEL) {
  if (!url) {
    console.warn(
      "[frontend] API_BASE_URL is not set.\n" +
        "Vercel hosts the POS UI only. Catalog/cart will stay empty until you set\n" +
        "API_BASE_URL=https://YOUR-BACKEND-DOMAIN (no trailing slash) and redeploy.\n" +
        "Do not set it to this Vercel URL.",
    );
  } else {
    const host = hostnameOf(url);
    if (!host) {
      console.error(`[frontend] API_BASE_URL is not a valid absolute URL: ${url}`);
      process.exit(1);
    }
    if (isVercelHost(host) || isPlaceholder(url)) {
      console.error(
        "[frontend] API_BASE_URL must be the public HTTPS origin of the GPU FastAPI backend.\n" +
          "It cannot be a Vercel URL or a placeholder like https://YOUR-BACKEND-DOMAIN.",
      );
      process.exit(1);
    }
  }
}

const dest = path.join(__dirname, "..", "src", "config.js");
const contents = `window.RETAIL_VISION_API_URL = ${JSON.stringify(url)};\n`;
fs.writeFileSync(dest, contents);
console.log(`[frontend] wrote ${path.relative(process.cwd(), dest)} API_BASE_URL=${url || "(same origin)"}`);
