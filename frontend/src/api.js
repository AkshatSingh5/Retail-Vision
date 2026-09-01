(function (global) {
  const STORAGE_KEY = "RETAIL_VISION_API_URL";

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

  function isVercelHost(host) {
    return host === "vercel.app" || host.endsWith(".vercel.app");
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

  function readApiBase() {
    const candidates = [readQueryOverride(), readBakedIn(), readStored()];
    for (const url of candidates) {
      if (isUsableBackend(url)) return url;
    }
    return "";
  }

  const API_BASE_URL = readApiBase();

  function isRemoteStaticHost() {
    const host = ((global.location && global.location.hostname) || "").toLowerCase();
    return isVercelHost(host);
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
