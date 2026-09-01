(function (global) {
  const STORAGE_KEY = "RETAIL_VISION_API_URL";
  const LOCAL_BACKEND = "http://127.0.0.1:8000";
  const LOCAL_UI_PORTS = new Set(["3000", "4173", "5173", "5500", "8080"]);

  function normalize(url) {
    return String(url || "").trim().replace(/\/$/, "");
  }

  function hostnameOf(url) {
    try {
      return new URL(url).hostname.toLowerCase();
    } catch (_e) {
      return "";
    }
  }

  function pageHost() {
    return ((global.location && global.location.hostname) || "").toLowerCase();
  }

  function pagePort() {
    return String((global.location && global.location.port) || "");
  }

  function isVercelHost(host) {
    return host === "vercel.app" || host.endsWith(".vercel.app");
  }

  function isStaticFrontendHost() {
    const host = pageHost();
    if (!host) return false;
    if (isVercelHost(host)) return true;
    if (host.endsWith(".netlify.app") || host.endsWith(".github.io") || host.endsWith(".pages.dev")) {
      return true;
    }
    if ((host === "localhost" || host === "127.0.0.1") && LOCAL_UI_PORTS.has(pagePort())) {
      return true;
    }
    return false;
  }

  function isUsableBackend(url) {
    if (!url || !/^https?:\/\//i.test(url)) return false;
    const host = hostnameOf(url);
    if (!host) return false;
    // The Vercel project is the static UI. Pointing the API at it yields /cart 404s.
    if (isVercelHost(host)) return false;
    return true;
  }

  function readQueryOverride() {
    try {
      const raw = new URLSearchParams(global.location.search).get("api");
      if (!raw) return "";
      const url = normalize(raw);
      if (!isUsableBackend(url)) return "";
      try {
        global.localStorage.setItem(STORAGE_KEY, url);
      } catch (_e) {
        /* private mode */
      }
      return url;
    } catch (_e) {
      return "";
    }
  }

  function readStored() {
    try {
      return normalize(global.localStorage.getItem(STORAGE_KEY));
    } catch (_e) {
      return "";
    }
  }

  function readBakedIn() {
    const fromWindow =
      typeof global.RETAIL_VISION_API_URL === "string" ? global.RETAIL_VISION_API_URL : "";
    return normalize(fromWindow);
  }

  function readLocalBackendFallback() {
    const host = pageHost();
    if (host !== "localhost" && host !== "127.0.0.1") return "";
    if (global.location && global.location.protocol === "https:") return "";
    const port = pagePort();
    // FastAPI already serves the UI on 8000 — keep same-origin relative URLs.
    if (!port || port === "8000") return "";
    if (LOCAL_UI_PORTS.has(port) || /^\d+$/.test(port)) return LOCAL_BACKEND;
    return "";
  }

  function readApiBase() {
    const candidates = [readQueryOverride(), readBakedIn(), readStored(), readLocalBackendFallback()];
    for (const url of candidates) {
      if (isUsableBackend(url)) return url;
    }
    return "";
  }

  const API_BASE_URL = readApiBase();

  function isRemoteStaticHost() {
    return isStaticFrontendHost();
  }

  function isConfigured() {
    return Boolean(API_BASE_URL) || !isRemoteStaticHost();
  }

  function apiUrl(path) {
    if (!path) return API_BASE_URL || "";
    if (/^https?:\/\//i.test(path)) return path;
    const normalized = path.startsWith("/") ? path : `/${path}`;
    return API_BASE_URL ? `${API_BASE_URL}${normalized}` : normalized;
  }

  function mediaUrl(path) {
    if (!path) return path;
    if (/^https?:\/\//i.test(path) || path.startsWith("blob:") || path.startsWith("data:")) {
      return path;
    }
    return apiUrl(path);
  }

  global.RetailVisionAPI = { API_BASE_URL, apiUrl, mediaUrl, isConfigured, isRemoteStaticHost };
})(window);
