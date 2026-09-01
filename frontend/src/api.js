(function (global) {
  function readApiBase() {
    const fromWindow =
      typeof global.RETAIL_VISION_API_URL === "string" ? global.RETAIL_VISION_API_URL.trim() : "";
    return fromWindow.replace(/\/$/, "");
  }

  const API_BASE_URL = readApiBase();

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

  global.RetailVisionAPI = { API_BASE_URL, apiUrl, mediaUrl };
})(window);
