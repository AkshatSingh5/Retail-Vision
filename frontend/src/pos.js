/**
 * Retail Vision POS Terminal — Frontend Application Logic
 *
 * Integrated with the complete YOLO26m + DINOv2 + pgvector recognition pipeline,
 * active cart synchronization, and PDF invoice generation.
 */

(function () {
  "use strict";

  // =========================================================================
  // CONSTANTS & STATE ENUMS
  // =========================================================================
  const rupee = (value) => `₹${Number(value ?? 0).toLocaleString("en-IN")}`;

  const POS_STATES = {
    CAMERA_OFFLINE: "CAMERA_OFFLINE",
    CAMERA_STARTING: "CAMERA_STARTING",
    CAMERA_ONLINE: "CAMERA_ONLINE",
    CAPTURING: "CAPTURING",
    SCANNING: "SCANNING",
    PRODUCT_DETECTED: "PRODUCT_DETECTED",
    PRODUCT_RECOGNIZED: "PRODUCT_RECOGNIZED",
    NO_PRODUCT: "NO_PRODUCT",
    MULTIPLE_PRODUCTS: "MULTIPLE_PRODUCTS",
    LOW_QUALITY: "LOW_QUALITY",
    UNKNOWN_PRODUCT: "UNKNOWN_PRODUCT",
    AMBIGUOUS_MATCH: "AMBIGUOUS_MATCH",
    ERROR: "ERROR",
  };

  // =========================================================================
  // DOM ELEMENT SELECTORS
  // =========================================================================
  const els = {
    // Header & Status
    camStatusPill: document.getElementById("cam-status-pill"),
    camStatusText: document.getElementById("cam-status-text"),
    txnId: document.getElementById("txn-id"),
    btnOpenCatalog: document.getElementById("btn-open-catalog"),
    btnOpenCaption: document.getElementById("btn-open-caption"),
    btnNewTxn: document.getElementById("btn-new-txn"),
    viewportCamLabel: document.getElementById("viewport-cam-label"),
    scannerStatusLine: document.getElementById("scanner-status-line"),
    btnScanNext: document.getElementById("btn-scan-next"),
    detectionShowcasePanel: document.getElementById("detection-showcase-panel"),

    // Viewfinder & Camera
    viewfinder: document.getElementById("viewfinder"),
    liveVideo: document.getElementById("live-video"),
    liveFeed: document.getElementById("live-feed"),
    overlayCanvas: document.getElementById("overlay-canvas"),
    cameraLoadingOverlay: document.getElementById("camera-loading-overlay"),
    cameraLoadingText: document.getElementById("camera-loading-text"),
    cameraOfflineOverlay: document.getElementById("camera-offline-overlay"),
    btnQuickStart: document.getElementById("btn-quick-start"),
    feedLiveBadge: document.getElementById("feed-live-badge"),
    fpsMetaTag: document.getElementById("fps-meta-tag"),
    fpsValue: document.getElementById("fps-value"),
    latencyMetaTag: document.getElementById("latency-meta-tag"),
    latencyValue: document.getElementById("latency-value"),
    detectionStatusChip: document.getElementById("detection-status-chip"),
    statusChipDot: document.getElementById("status-chip-dot"),
    statusChipText: document.getElementById("status-chip-text"),
    hudReticle: document.getElementById("hud-reticle"),

    // Toolbar
    btnCameraToggle: document.getElementById("btn-camera-toggle"),
    btnCameraText: document.getElementById("btn-camera-text"),
    btnCaptureSnapshot: document.getElementById("btn-capture-snapshot"),
    btnScanProduct: document.getElementById("btn-scan-product"),
    scanBtnLabel: document.getElementById("scan-btn-label"),
    visionMessageBar: document.getElementById("vision-message-bar"),
    visionMessageIcon: document.getElementById("vision-message-icon"),
    visionMessageText: document.getElementById("vision-message-text"),

    // Showcase Panel
    showcaseIdle: document.getElementById("showcase-idle"),
    showcaseFound: document.getElementById("showcase-found"),
    detectedProdImg: document.getElementById("detected-prod-img"),
    detectedProdCategory: document.getElementById("detected-prod-category"),
    detectedProdConfidenceBadge: document.getElementById("detected-prod-confidence-badge"),
    detectedProdName: document.getElementById("detected-prod-name"),
    detectedProdSku: document.getElementById("detected-prod-sku"),
    detectedProdWeightWrap: document.getElementById("detected-prod-weight-wrap"),
    detectedProdWeight: document.getElementById("detected-prod-weight"),
    detectedProdPrice: document.getElementById("detected-prod-price"),
    btnAddDetected: document.getElementById("btn-add-detected"),
    btnScanAgainFound: document.getElementById("btn-scan-again-found"),

    showcaseFailed: document.getElementById("showcase-failed"),
    failedCropImg: document.getElementById("failed-crop-img"),
    failedBadgeTag: document.getElementById("failed-badge-tag"),
    failedStatusTitle: document.getElementById("failed-status-title"),
    failedTitleText: document.getElementById("failed-title-text"),
    failedMsgText: document.getElementById("failed-msg-text"),
    btnTryScanAgain: document.getElementById("btn-try-scan-again"),
    btnRegisterUnknown: document.getElementById("btn-register-unknown"),

    // Cart & Billing
    cartItemBadge: document.getElementById("cart-item-badge"),
    btnClearCart: document.getElementById("btn-clear-cart"),
    cartAlerts: document.getElementById("cart-alerts"),
    cartEmptyState: document.getElementById("cart-empty-state"),
    cartLinesList: document.getElementById("cart-lines-list"),
    discountInput: document.getElementById("discount-input"),
    discountPresetChips: document.querySelectorAll(".discount-chip"),
    discountRow: document.getElementById("discount-row"),
    billSubtotal: document.getElementById("bill-subtotal"),
    billTax: document.getElementById("bill-tax"),
    billDiscount: document.getElementById("bill-discount"),
    billGrandTotal: document.getElementById("bill-grand-total"),
    btnGenerateBill: document.getElementById("btn-generate-bill"),
    checkoutBtnText: document.getElementById("checkout-btn-text"),

    // Modals
    checkoutModal: document.getElementById("checkout-modal"),
    modalInvoiceNum: document.getElementById("modal-invoice-num"),
    modalInvoiceTotal: document.getElementById("modal-invoice-total"),
    modalPdfLink: document.getElementById("modal-pdf-link"),
    btnModalNewTxn: document.getElementById("btn-modal-new-txn"),

    productRegModal: document.getElementById("product-reg-modal"),
    btnCloseReg: document.getElementById("btn-close-reg"),
    btnCancelReg: document.getElementById("btn-cancel-reg"),
    regModalStatus: document.getElementById("reg-modal-status"),
    productRegForm: document.getElementById("product-reg-form"),
    regImgPreview: document.getElementById("reg-img-preview"),
    btnRegRetake: document.getElementById("btn-reg-retake"),
    regFileInput: document.getElementById("reg-file-input"),

    similarModal: document.getElementById("similar-modal"),
    similarPreviewImg: document.getElementById("similar-preview-img"),
    similarProdName: document.getElementById("similar-prod-name"),
    similarProdPrice: document.getElementById("similar-prod-price"),
    similarProdSku: document.getElementById("similar-prod-sku"),
    btnUseSimilar: document.getElementById("btn-use-similar"),
    btnCreateForce: document.getElementById("btn-create-force"),

    catalogModal: document.getElementById("catalog-modal"),
    btnCloseCatalog: document.getElementById("btn-close-catalog"),
    catalogSearchInput: document.getElementById("catalog-search-input"),
    catalogTableBody: document.getElementById("catalog-table-body"),

    captionModal: document.getElementById("caption-modal"),
    btnCloseCaption: document.getElementById("btn-close-caption"),
    captionFileInput: document.getElementById("caption-file-input"),
    captionImgPreview: document.getElementById("caption-img-preview"),
    btnGenerateCaption: document.getElementById("btn-generate-caption"),
    captionResultText: document.getElementById("caption-result-text"),
    btnCopyPrompt: document.getElementById("btn-copy-prompt"),
    captionStatusIndicator: document.getElementById("caption-status-indicator"),
  };

  // =========================================================================
  // APPLICATION STATE
  // =========================================================================
  let currentState = POS_STATES.CAMERA_OFFLINE;
  let isApiBusy = false;
  let catalogData = [];
  let currentScanResult = null;
  let activeCartItemCount = 0;
  let currentSimilarProductId = null;
  let lastCapturedBlob = null;
  let scanAnimationTimer = null;

  // Browser Camera Stream State
  const browserCamera = {
    active: false,
    stream: null,
  };
  let userWantsCamera = false;
  let cameraToggleLock = false;

  // Backend Vision Hub Status Cache
  let hubVisionStatus = {
    running: false,
    camera_active: false,
    detection_active: false,
    fps: 0,
    latency_ms: 0,
    error: null,
  };

  // =========================================================================
  // HELPER UTILITIES
  // =========================================================================
  function apiUrl(path) {
    return window.RetailVisionAPI ? window.RetailVisionAPI.apiUrl(path) : path;
  }

  function mediaUrl(path) {
    if (!path) return path;
    return window.RetailVisionAPI ? window.RetailVisionAPI.mediaUrl(path) : path;
  }

  function backendIsConfigured() {
    return !window.RetailVisionAPI || window.RetailVisionAPI.isConfigured();
  }

  let backendUnavailable = false;
  let pollTimers = [];

  function stopBackendPolling() {
    pollTimers.forEach((id) => clearInterval(id));
    pollTimers = [];
  }

  function markBackendUnavailable(err) {
    if (backendUnavailable) return;
    backendUnavailable = true;
    stopBackendPolling();
    const missing = window.RetailVisionAPI && window.RetailVisionAPI.isRemoteStaticHost() && !window.RetailVisionAPI.API_BASE_URL;
    const detail = err && err.message ? String(err.message) : "";
    const msg = missing
      ? "This Vercel UI has no backend URL. Set API_BASE_URL to your FastAPI origin (not this Vercel URL) and redeploy."
      : `Cannot reach the POS API${detail ? ` (${detail})` : ""}.`;
    showVisionMessage(msg, "error");
  }

  function isCameraActive() {
    return Boolean(browserCamera.active || hubVisionStatus.camera_active);
  }

  async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (!(options.body instanceof FormData) && !headers["Content-Type"] && options.body) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(apiUrl(path), { headers, ...options });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_e) {
        /* ignore parse error */
      }
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.status = response.status;
      throw error;
    }
    if (response.status === 204) return null;
    const type = response.headers.get("content-type") || "";
    if (type.includes("application/json")) return response.json();
    return response;
  }

  function showVisionMessage(text, type = "info") {
    if (!text) {
      els.visionMessageBar.classList.add("hidden");
      return;
    }
    els.visionMessageBar.classList.remove("hidden");
    els.visionMessageText.textContent = text;
    els.visionMessageIcon.textContent = type === "error" ? "⚠️" : type === "success" ? "✓" : "ℹ️";
  }

  // =========================================================================
  // CANVAS BOUNDING BOX & SCAN RETICLE VISUALIZATION
  // =========================================================================
  function resizeCanvasToDisplaySize() {
    const canvas = els.overlayCanvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
  }

  function clearCanvasOverlay() {
    const canvas = els.overlayCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function startScanningAnimation() {
    stopScanningAnimation();
    resizeCanvasToDisplaySize();
    const canvas = els.overlayCanvas;
    const ctx = canvas.getContext("2d");
    let lineY = 0;
    let direction = 1;

    function step() {
      if (currentState !== POS_STATES.SCANNING) {
        clearCanvasOverlay();
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw horizontal glowing scan line
      const gradient = ctx.createLinearGradient(0, lineY - 15, 0, lineY + 15);
      gradient.addColorStop(0, "rgba(245, 158, 11, 0)");
      gradient.addColorStop(0.5, "rgba(245, 158, 11, 0.75)");
      gradient.addColorStop(1, "rgba(245, 158, 11, 0)");

      ctx.fillStyle = gradient;
      ctx.fillRect(0, lineY - 15, canvas.width, 30);

      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, lineY);
      ctx.lineTo(canvas.width, lineY);
      ctx.stroke();

      lineY += direction * 5;
      if (lineY > canvas.height) {
        lineY = canvas.height;
        direction = -1;
      } else if (lineY < 0) {
        lineY = 0;
        direction = 1;
      }

      scanAnimationTimer = requestAnimationFrame(step);
    }

    scanAnimationTimer = requestAnimationFrame(step);
  }

  function stopScanningAnimation() {
    if (scanAnimationTimer) {
      cancelAnimationFrame(scanAnimationTimer);
      scanAnimationTimer = null;
    }
  }

  function drawBoundingBox(bbox, labelText, isSuccess = true) {
    if (!bbox || bbox.length < 4) return;
    resizeCanvasToDisplaySize();
    const canvas = els.overlayCanvas;
    const ctx = canvas.getContext("2d");
    clearCanvasOverlay();

    // Source coordinates from frame (assume standard 640x480 or 1280x720 video)
    // Scale coordinates proportionally to displayed canvas
    const [x1, y1, x2, y2] = bbox;
    
    // Determine scaling ratio
    // If bbox is already normalized (0..1), scale by canvas width/height
    let bx, by, bw, bh;
    if (x2 <= 1.0 && y2 <= 1.0) {
      bx = x1 * canvas.width;
      by = y1 * canvas.height;
      bw = (x2 - x1) * canvas.width;
      bh = (y2 - y1) * canvas.height;
    } else {
      // Estimate frame scale based on typical camera frame aspect
      const baseW = 1280;
      const baseH = 720;
      const scaleX = canvas.width / baseW;
      const scaleY = canvas.height / baseH;
      bx = x1 * scaleX;
      by = y1 * scaleY;
      bw = (x2 - x1) * scaleX;
      bh = (y2 - y1) * scaleY;
    }

    // Clamp coordinates
    bx = Math.max(10, Math.min(canvas.width - 20, bx));
    by = Math.max(10, Math.min(canvas.height - 20, by));
    bw = Math.max(40, Math.min(canvas.width - bx - 10, bw));
    bh = Math.max(40, Math.min(canvas.height - by - 10, bh));

    const color = isSuccess ? "#10b981" : "#f59e0b";
    const bgBadge = isSuccess ? "rgba(16, 185, 129, 0.9)" : "rgba(245, 158, 11, 0.9)";

    // Draw box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.strokeRect(bx, by, bw, bh);

    // Draw corner accents
    const cLen = 14;
    ctx.lineWidth = 5;
    ctx.beginPath();
    // Top Left
    ctx.moveTo(bx, by + cLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cLen, by);
    // Top Right
    ctx.moveTo(bx + bw - cLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cLen);
    // Bottom Left
    ctx.moveTo(bx, by + bh - cLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cLen, by + bh);
    // Bottom Right
    ctx.moveTo(bx + bw - cLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cLen);
    ctx.stroke();

    // Draw Label Badge
    if (labelText) {
      ctx.shadowBlur = 0;
      ctx.font = "bold 12px Inter, sans-serif";
      const textWidth = ctx.measureText(labelText).width;
      const badgeH = 24;
      const badgeW = textWidth + 16;
      const badgeX = bx;
      const badgeY = Math.max(0, by - badgeH - 4);

      ctx.fillStyle = bgBadge;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4) : ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
      ctx.fill();

      ctx.fillStyle = "#090c10";
      ctx.fillText(labelText, badgeX + 8, badgeY + 16);
    }
  }

  // =========================================================================
  // STATE MACHINE & UI UPDATE HANDLER
  // =========================================================================
  function setPosState(nextState, message = "") {
    currentState = nextState;
    if (message) showVisionMessage(message);

    const cameraOn = isCameraActive() || (userWantsCamera && currentState === POS_STATES.CAMERA_STARTING);
    const busy = isApiBusy || [POS_STATES.CAPTURING, POS_STATES.SCANNING].includes(currentState);

    // Close Camera stays clickable during a scan; Open Camera waits until idle.
    els.btnCameraToggle.disabled = cameraToggleLock || (busy && !cameraOn);
    els.btnCameraText.textContent = cameraOn ? "Close Camera" : "Open Camera";
    els.btnCameraToggle.classList.toggle("is-close", cameraOn);
    els.btnCameraToggle.setAttribute("aria-pressed", cameraOn ? "true" : "false");
    els.btnCaptureSnapshot.disabled = busy || !cameraOn;
    els.btnScanProduct.disabled = busy || !cameraOn;
    els.btnGenerateBill.disabled = busy || activeCartItemCount === 0;

    // Viewfinder Status Chip Text & Style
    const statusMap = {
      [POS_STATES.CAMERA_OFFLINE]: { label: "Camera Offline", class: "failed" },
      [POS_STATES.CAMERA_STARTING]: { label: "Starting Camera...", class: "scanning" },
      [POS_STATES.CAMERA_ONLINE]: { label: "Camera Online · Ready to Scan", class: "ready" },
      [POS_STATES.CAPTURING]: { label: "Capturing Image Frame...", class: "scanning" },
      [POS_STATES.SCANNING]: { label: "Detecting & Identifying Product...", class: "scanning" },
      [POS_STATES.PRODUCT_DETECTED]: { label: "Product Detected", class: "ready" },
      [POS_STATES.PRODUCT_RECOGNIZED]: { label: "Product Recognized ✓", class: "found" },
      [POS_STATES.NO_PRODUCT]: { label: "No Product Detected", class: "failed" },
      [POS_STATES.MULTIPLE_PRODUCTS]: { label: "Multiple Products Detected", class: "failed" },
      [POS_STATES.LOW_QUALITY]: { label: "Image Quality Too Low", class: "failed" },
      [POS_STATES.UNKNOWN_PRODUCT]: { label: "Unknown Product", class: "failed" },
      [POS_STATES.AMBIGUOUS_MATCH]: { label: "Ambiguous Match", class: "failed" },
      [POS_STATES.ERROR]: { label: "Scanner Error", class: "failed" },
    };

    const currentConfig = statusMap[currentState] || { label: "Ready", class: "ready" };
    els.statusChipText.textContent = currentConfig.label;
    els.detectionStatusChip.className = `status-chip ${currentConfig.class}`;

    // Camera Status Pill in Top Header
    els.camStatusPill.classList.toggle("online", cameraOn);
    els.camStatusPill.classList.toggle("offline", !cameraOn);
    els.camStatusText.textContent = cameraOn ? "online" : "offline";

    if (els.viewfinder) {
      els.viewfinder.classList.toggle("is-live", cameraOn);
    }

    if (els.viewportCamLabel) {
      els.viewportCamLabel.textContent = cameraOn ? "camera on." : "camera off.";
    }

    if (els.scannerStatusLine) {
      if (currentState === POS_STATES.CAMERA_STARTING) {
        els.scannerStatusLine.textContent = "Camera Offline Status: Starting camera";
      } else if (!cameraOn) {
        els.scannerStatusLine.textContent = "Camera Offline Status: Camera closed";
      } else if (currentState === POS_STATES.SCANNING || currentState === POS_STATES.CAPTURING) {
        els.scannerStatusLine.textContent = "Camera Online Status: Scanning";
      } else {
        els.scannerStatusLine.textContent = "Camera Online Status: Camera live";
      }
    }

    // Scan Button text & spinner
    if (currentState === POS_STATES.SCANNING) {
      els.scanBtnLabel.textContent = "Identifying Product...";
      startScanningAnimation();
    } else {
      els.scanBtnLabel.textContent = "Scan Product";
      stopScanningAnimation();
    }
  }

  // =========================================================================
  // CAMERA STREAM CONTROLLERS (HUB STREAM + BROWSER FALLBACK)
  // =========================================================================
  function setCameraLoadingOverlay(visible, text = "Starting camera stream...") {
    els.cameraLoadingText.textContent = text;
    els.cameraLoadingOverlay.classList.toggle("hidden", !visible);
  }

  function stopBrowserCameraStream() {
    if (browserCamera.stream) {
      browserCamera.stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_e) {
          /* ignore */
        }
      });
      browserCamera.stream = null;
    }
    browserCamera.active = false;
    if (els.liveVideo) {
      try {
        els.liveVideo.pause();
      } catch (_e) {
        /* ignore */
      }
      els.liveVideo.srcObject = null;
      els.liveVideo.removeAttribute("src");
      els.liveVideo.classList.add("hidden");
    }
  }

  function stopHubPreview() {
    hubVisionStatus.camera_active = false;
    if (els.liveFeed) {
      els.liveFeed.removeAttribute("src");
      els.liveFeed.src = "";
      els.liveFeed.classList.add("hidden");
    }
  }

  async function stopHubCamera() {
    try {
      await api("/pos/camera/stop", { method: "POST" });
    } catch (_e) {
      /* Hub may already be off, or the static UI has no backend. */
    }
    hubVisionStatus.camera_active = false;
  }

  async function startBrowserCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("WebRTC camera not supported in this browser.");
    }
    stopBrowserCameraStream();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    if (!userWantsCamera) {
      stream.getTracks().forEach((track) => track.stop());
      throw new Error("Camera open cancelled.");
    }
    browserCamera.stream = stream;
    browserCamera.active = true;
    els.liveVideo.srcObject = stream;
    els.liveVideo.classList.remove("hidden");
    els.liveFeed.classList.add("hidden");
    els.cameraOfflineOverlay.classList.add("hidden");
    els.feedLiveBadge.classList.remove("hidden");
    try {
      await els.liveVideo.play();
    } catch (_e) {
      /* autoplay can be blocked; srcObject still shows frames after a tap */
    }
    return stream;
  }

  async function closeCamera() {
    userWantsCamera = false;
    setCameraLoadingOverlay(true, "Closing camera...");
    stopBrowserCameraStream();
    stopHubPreview();
    els.cameraOfflineOverlay.classList.remove("hidden");
    els.feedLiveBadge.classList.add("hidden");
    els.fpsMetaTag.classList.add("hidden");
    els.latencyMetaTag.classList.add("hidden");
    clearCanvasOverlay();
    setCameraLoadingOverlay(false);
    setPosState(POS_STATES.CAMERA_OFFLINE, "Camera turned off.");
    stopHubCamera();
  }

  async function openCamera() {
    userWantsCamera = true;
    setCameraLoadingOverlay(true, "Connecting to camera feed...");
    setPosState(POS_STATES.CAMERA_STARTING, "Opening camera...");

    try {
      await startBrowserCamera();
      if (!userWantsCamera) {
        stopBrowserCameraStream();
        return;
      }
      setCameraLoadingOverlay(false);
      setPosState(POS_STATES.CAMERA_ONLINE, "Camera online. Place product in scanning zone.");
      return;
    } catch (browserErr) {
      if (!userWantsCamera) return;
      console.warn("Browser camera unavailable, falling back to backend camera hub:", browserErr);
    }

    if (!userWantsCamera) return;

    try {
      const info = await api("/pos/camera/start", { method: "POST" });
      if (!userWantsCamera) {
        await stopHubCamera();
        stopHubPreview();
        return;
      }
      hubVisionStatus = { ...hubVisionStatus, ...info };
      els.liveFeed.src = `${apiUrl("/pos/stream")}?t=${Date.now()}`;
      els.liveFeed.classList.remove("hidden");
      els.cameraOfflineOverlay.classList.add("hidden");
      els.feedLiveBadge.classList.remove("hidden");
      setCameraLoadingOverlay(false);
      setPosState(POS_STATES.CAMERA_ONLINE, "Camera online via Retail Vision Hub.");
    } catch (hubErr) {
      userWantsCamera = false;
      setCameraLoadingOverlay(false);
      els.cameraOfflineOverlay.classList.remove("hidden");
      setPosState(POS_STATES.ERROR, hubErr.message || "Could not open camera stream.");
    }
  }

  async function toggleCamera() {
    const shouldClose =
      userWantsCamera || isCameraActive() || currentState === POS_STATES.CAMERA_STARTING;

    if (shouldClose) {
      userWantsCamera = false;
      stopBrowserCameraStream();
      stopHubPreview();
      if (cameraToggleLock && currentState === POS_STATES.CAMERA_STARTING) {
        setCameraLoadingOverlay(false);
        els.cameraOfflineOverlay.classList.remove("hidden");
        els.feedLiveBadge.classList.add("hidden");
        setPosState(POS_STATES.CAMERA_OFFLINE, "Camera turned off.");
        stopHubCamera();
        return;
      }
      cameraToggleLock = true;
      els.btnCameraToggle.disabled = true;
      try {
        await closeCamera();
      } finally {
        cameraToggleLock = false;
        setPosState(currentState);
      }
      return;
    }

    if (cameraToggleLock) return;
    cameraToggleLock = true;
    els.btnCameraToggle.disabled = true;
    try {
      await openCamera();
    } finally {
      cameraToggleLock = false;
      setPosState(currentState);
    }
  }

  async function captureCurrentFrameBlob() {
    if (browserCamera.active && els.liveVideo) {
      const video = els.liveVideo;
      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;
      const canvas = document.createElement("canvas");
      canvas.width = vw;
      canvas.height = vh;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, vw, vh);
      return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
          if (!blob) reject(new Error("Frame capture failed."));
          else resolve(blob);
        }, "image/jpeg", 0.92);
      });
    }

    // Backend camera hub snapshot
    const response = await fetch(apiUrl("/pos/camera/capture"), { method: "POST" });
    if (!response.ok) throw new Error("Backend camera capture failed.");
    return await response.blob();
  }

  // =========================================================================
  // PRODUCT SCANNING & RECOGNITION PIPELINE
  // =========================================================================
  function setShowcaseIdle(idle) {
    if (els.detectionShowcasePanel) {
      els.detectionShowcasePanel.classList.toggle("is-idle", idle);
    }
  }

  function renderFoundProductCard(product, bbox) {
    setShowcaseIdle(false);
    els.showcaseIdle.classList.add("hidden");
    els.showcaseFailed.classList.add("hidden");
    els.showcaseFound.classList.remove("hidden");

    // Fill Product Data
    els.detectedProdName.textContent = product.name || "Unknown Product";
    els.detectedProdSku.textContent = product.sku || "—";
    els.detectedProdPrice.textContent = Number(product.price || 0).toLocaleString("en-IN");
    
    if (product.category) {
      els.detectedProdCategory.textContent = product.category.toUpperCase();
      els.detectedProdCategory.classList.remove("hidden");
    } else {
      els.detectedProdCategory.classList.add("hidden");
    }

    if (product.weight) {
      els.detectedProdWeight.textContent = product.weight;
      els.detectedProdWeightWrap.classList.remove("hidden");
    } else {
      els.detectedProdWeightWrap.classList.add("hidden");
    }

    const confidencePct = Math.round(Number(product.confidence || 0) * 100);
    els.detectedProdConfidenceBadge.textContent = `${confidencePct}% Confidence`;

    // Crop or Product Image
    const imgUrl = product.image_url || (currentScanResult && currentScanResult.preview_url);
    if (imgUrl) {
      els.detectedProdImg.src = `${mediaUrl(imgUrl)}?t=${Date.now()}`;
      els.detectedProdImg.style.display = "block";
    } else {
      els.detectedProdImg.style.display = "none";
    }

    // Draw visual bounding box
    drawBoundingBox(bbox, `${product.name} (${confidencePct}%)`, true);
  }

  function renderFailedRecognitionCard(scanResponse, bbox) {
    setShowcaseIdle(false);
    els.showcaseIdle.classList.add("hidden");
    els.showcaseFound.classList.add("hidden");
    els.showcaseFailed.classList.remove("hidden");

    const status = scanResponse.status || "unknown";
    let title = "Unknown Product";
    let msg = "Unable to confidently identify this product against our catalog.";
    let badgeText = "UNKNOWN";
    let allowRegister = true;

    if (status === "no_product") {
      title = "No Product Detected";
      msg = "Please place one product clearly inside the scanning zone.";
      badgeText = "NO ITEM";
      allowRegister = false;
    } else if (status === "multiple_products") {
      title = "Multiple Products Detected";
      msg = "Please place only one product inside the camera area at a time.";
      badgeText = "MULTIPLE";
      allowRegister = false;
    } else if (status === "low_image_quality") {
      title = "Image Quality Too Low";
      msg = scanResponse.message || "Please hold the product steady and scan again.";
      badgeText = "BLURRY";
      allowRegister = false;
    } else if (status === "ambiguous") {
      title = "Ambiguous Match";
      msg = "Multiple visually similar items detected in database. Please scan again.";
      badgeText = "AMBIGUOUS";
      allowRegister = false;
    }

    els.failedTitleText.textContent = title;
    els.failedMsgText.textContent = msg;
    els.failedBadgeTag.textContent = badgeText;
    els.btnRegisterUnknown.classList.toggle("hidden", !allowRegister);

    // Preview Crop if provided
    if (scanResponse.preview_url) {
      els.failedCropImg.src = `${mediaUrl(scanResponse.preview_url)}?t=${Date.now()}`;
      els.failedCropImg.style.display = "block";
    } else {
      els.failedCropImg.style.display = "none";
    }

    if (bbox) {
      drawBoundingBox(bbox, title, false);
    } else {
      clearCanvasOverlay();
    }
  }

  function resetShowcaseToIdle() {
    clearCanvasOverlay();
    els.showcaseFound.classList.add("hidden");
    els.showcaseFailed.classList.add("hidden");
    els.showcaseIdle.classList.add("hidden");
    setShowcaseIdle(true);
    currentScanResult = null;
    setPosState(isCameraActive() ? POS_STATES.CAMERA_ONLINE : POS_STATES.CAMERA_OFFLINE, "Ready to scan.");
  }

  async function executeProductScan() {
    if (!isCameraActive()) {
      setPosState(POS_STATES.ERROR, "Please open the camera before scanning.");
      return;
    }
    if (isApiBusy) return;

    isApiBusy = true;
    clearCanvasOverlay();
    setPosState(POS_STATES.SCANNING, "Capturing frame and querying vector gallery...");

    try {
      const formData = new FormData();
      if (browserCamera.active) {
        const blob = await captureCurrentFrameBlob();
        lastCapturedBlob = blob;
        formData.append("use_camera", "false");
        formData.append("image", blob, "scan.jpg");
      } else {
        formData.append("use_camera", "true");
      }

      const response = await api("/products/scan", {
        method: "POST",
        body: formData,
      });

      currentScanResult = response;
      const bbox = response.items && response.items[0] && response.items[0].bbox;

      if ((response.found || response.status === "found") && response.product) {
        setPosState(POS_STATES.PRODUCT_RECOGNIZED, `Recognized: ${response.product.name}`);
        renderFoundProductCard(response.product, bbox);
      } else {
        const stateMap = {
          no_product: POS_STATES.NO_PRODUCT,
          multiple_products: POS_STATES.MULTIPLE_PRODUCTS,
          low_image_quality: POS_STATES.LOW_QUALITY,
          ambiguous: POS_STATES.AMBIGUOUS_MATCH,
        };
        setPosState(stateMap[response.status] || POS_STATES.UNKNOWN_PRODUCT, response.message || "Unknown Product.");
        renderFailedRecognitionCard(response, bbox);
      }
    } catch (err) {
      console.error("Scan error:", err);
      setPosState(POS_STATES.ERROR, err.message || "Product recognition failed.");
      showVisionMessage(err.message || "Recognition service unavailable.", "error");
    } finally {
      isApiBusy = false;
      stopScanningAnimation();
    }
  }

  async function addDetectedProductToCart() {
    if (!currentScanResult || !currentScanResult.product) return;
    const prod = currentScanResult.product;
    isApiBusy = true;
    try {
      await api("/cart/items", {
        method: "POST",
        body: JSON.stringify({
          product_id: prod.product_id || prod.id,
          quantity: 1,
        }),
      });
      await syncCart();
      showVisionMessage(`✓ Added ${prod.name} to cart.`, "success");
      resetShowcaseToIdle();
    } catch (err) {
      showVisionMessage(err.message || "Failed to add product to cart.", "error");
    } finally {
      isApiBusy = false;
    }
  }

  // =========================================================================
  // SHOPPING CART SYNCHRONIZATION & ACTIONS
  // =========================================================================
  function renderCartLines(cart) {
    activeCartItemCount = (cart.items || []).length;
    els.cartItemBadge.textContent = `${activeCartItemCount} ${activeCartItemCount === 1 ? "item" : "items"}`;
    els.txnId.textContent = cart.transaction_id || "—";

    // Summary Totals
    els.billSubtotal.textContent = rupee(cart.subtotal);
    els.billTax.textContent = rupee(cart.tax);
    els.billDiscount.textContent = rupee(cart.discount);
    els.billGrandTotal.textContent = rupee(cart.grand_total);

    // Keep discount input in sync
    if (document.activeElement !== els.discountInput) {
      els.discountInput.value = cart.discount_percent ?? 0;
    }
    els.discountPresetChips.forEach((chip) => {
      const pct = Number(chip.dataset.pct);
      chip.classList.toggle("active", pct === Number(cart.discount_percent || 0));
    });

    // Alert Banner
    const alerts = cart.alerts || [];
    if (alerts.length > 0) {
      els.cartAlerts.classList.remove("hidden");
      els.cartAlerts.innerHTML = alerts.map((a) => `<p>${a.message || a.reason}</p>`).join("");
    } else {
      els.cartAlerts.classList.add("hidden");
      els.cartAlerts.replaceChildren();
    }

    // Line Items View
    if (!cart.items || cart.items.length === 0) {
      els.cartEmptyState.classList.remove("hidden");
      els.cartLinesList.classList.add("hidden");
      els.cartLinesList.replaceChildren();
      els.btnGenerateBill.disabled = true;
      return;
    }

    els.cartEmptyState.classList.add("hidden");
    els.cartLinesList.classList.remove("hidden");
    els.cartLinesList.replaceChildren();
    els.btnGenerateBill.disabled = isApiBusy || [POS_STATES.CAPTURING, POS_STATES.SCANNING].includes(currentState);

    cart.items.forEach((item) => {
      const li = document.createElement("li");
      li.className = "cart-item-card";
      const weightBadge = item.weight ? `<span>· ${item.weight}</span>` : "";

      li.innerHTML = `
        <div class="item-main-info">
          <h4 class="item-name">${item.name}</h4>
          <div class="item-subtext">
            <span class="item-sku-tag">${item.sku}</span>
            ${weightBadge}
            <span>@ ${rupee(item.unit_price)}</span>
          </div>
        </div>
        <div class="item-actions-group">
          <div class="qty-control-box">
            <button class="qty-btn" data-act="dec" data-id="${item.product_id}" type="button" title="Decrease quantity">−</button>
            <span class="qty-val">${item.quantity}</span>
            <button class="qty-btn" data-act="inc" data-id="${item.product_id}" type="button" title="Increase quantity">+</button>
          </div>
          <div class="item-total-price">${rupee(item.total)}</div>
          <button class="item-remove-btn" data-act="del" data-id="${item.product_id}" type="button" title="Remove item">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          </button>
        </div>
      `;
      els.cartLinesList.appendChild(li);
    });
  }

  async function syncCart() {
    if (backendUnavailable) return;
    try {
      const cart = await api("/cart");
      renderCartLines(cart);
    } catch (err) {
      console.error("Cart sync failed:", err);
      if (err && (err.status === 404 || err.status === 0)) markBackendUnavailable(err);
    }
  }

  async function handleCartQuantityAction(productId, action) {
    isApiBusy = true;
    try {
      if (action === "inc") {
        await api(`/cart/items/${productId}/increase`, { method: "POST" });
      } else if (action === "dec") {
        await api(`/cart/items/${productId}/decrease`, { method: "POST" });
      } else if (action === "del") {
        await api(`/cart/items/${productId}`, { method: "DELETE" });
      }
      await syncCart();
    } catch (err) {
      showVisionMessage(err.message || "Cart update failed.", "error");
    } finally {
      isApiBusy = false;
    }
  }

  async function applyCartDiscount(percent) {
    try {
      await api("/cart/discount", {
        method: "POST",
        body: JSON.stringify({ percent: Number(percent || 0) }),
      });
      await syncCart();
    } catch (err) {
      showVisionMessage(err.message || "Discount could not be applied.", "error");
    }
  }

  // =========================================================================
  // CHECKOUT & PDF INVOICE GENERATION
  // =========================================================================
  async function executeCheckout() {
    if (activeCartItemCount === 0 || isApiBusy) return;
    isApiBusy = true;
    els.btnGenerateBill.disabled = true;
    els.checkoutBtnText.textContent = "GENERATING BILL...";

    try {
      const result = await api("/checkout", { method: "POST" });
      
      // Populate Checkout Success Modal
      els.modalInvoiceNum.textContent = result.invoice_number;
      els.modalInvoiceTotal.textContent = rupee(result.bill ? result.bill.grand_total : result.cart.grand_total);
      els.modalPdfLink.href = mediaUrl(result.pdf_url);

      // Show Modal
      els.checkoutModal.classList.remove("hidden");

      // Synchronize emptied cart from backend
      await syncCart();
      resetShowcaseToIdle();
      showVisionMessage(`✓ Bill generated: ${result.invoice_number}`, "success");
    } catch (err) {
      console.error("Checkout failed:", err);
      showVisionMessage(err.message || "Checkout failed. Please try again.", "error");
    } finally {
      isApiBusy = false;
      els.btnGenerateBill.disabled = false;
      els.checkoutBtnText.textContent = "GENERATE BILL";
    }
  }

  // =========================================================================
  // PRODUCT REGISTRATION MODAL CONTROLLERS
  // =========================================================================
  function openProductRegistrationModal() {
    els.regModalStatus.textContent = "";
    els.productRegForm.reset();
    document.getElementById("f-reg-tax").value = "18";

    // Set preview image if we have a captured crop or scan
    if (currentScanResult && currentScanResult.preview_url) {
      els.regImgPreview.src = `${mediaUrl(currentScanResult.preview_url)}?t=${Date.now()}`;
      els.regImgPreview.style.display = "block";
    } else if (lastCapturedBlob) {
      els.regImgPreview.src = URL.createObjectURL(lastCapturedBlob);
      els.regImgPreview.style.display = "block";
    } else {
      els.regImgPreview.style.display = "none";
    }

    els.productRegModal.classList.remove("hidden");
  }

  function closeProductRegistrationModal() {
    els.productRegModal.classList.add("hidden");
  }

  async function submitProductRegistration(forceCreate = false) {
    const form = els.productRegForm;
    const data = new FormData();
    data.append("name", document.getElementById("f-reg-name").value.trim());
    data.append("price", document.getElementById("f-reg-price").value);
    data.append("tax_rate", document.getElementById("f-reg-tax").value || "18");
    data.append("brand", document.getElementById("f-reg-brand").value.trim());
    data.append("category", document.getElementById("f-reg-category").value.trim());
    data.append("weight", document.getElementById("f-reg-weight").value.trim());
    
    const sku = document.getElementById("f-reg-sku").value.trim();
    if (sku) data.append("sku", sku);

    data.append("add_to_cart", "true");
    data.append("force_create", forceCreate ? "true" : "false");

    if (currentScanResult && currentScanResult.scan_id) {
      data.append("scan_id", currentScanResult.scan_id);
    }
    const file = els.regFileInput.files && els.regFileInput.files[0];
    if (file) {
      data.append("image", file);
    } else if (lastCapturedBlob) {
      data.append("image", lastCapturedBlob, "new_product.jpg");
    }

    const response = await fetch(apiUrl("/products/register"), {
      method: "POST",
      body: data,
    });

    const payload = await response.json().catch(() => ({}));

    // Handle Duplicate Similar Product check
    if (response.status === 409 && payload.detail && payload.detail.status === "similar_product_found") {
      showSimilarProductModal(payload.detail);
      throw new Error(payload.detail.message || "Similar product found in registry.");
    }

    if (!response.ok) {
      const detail = payload.detail;
      throw new Error(typeof detail === "string" ? detail : (detail && detail.message) || "Registration failed.");
    }

    closeProductRegistrationModal();
    await loadCatalog();
    await syncCart();
    resetShowcaseToIdle();
    showVisionMessage(`✓ Registered and added ${payload.name} to cart.`, "success");
  }

  function showSimilarProductModal(detail) {
    const product = detail.product || {};
    currentSimilarProductId = product.id;
    els.similarProdName.textContent = product.name || "Existing Product";
    els.similarProdPrice.textContent = rupee(product.price);
    els.similarProdSku.textContent = `SKU: ${product.sku || "—"}`;

    if (product.image_url) {
      els.similarPreviewImg.src = `${mediaUrl(product.image_url)}?t=${Date.now()}`;
      els.similarPreviewImg.style.display = "block";
    } else {
      els.similarPreviewImg.style.display = "none";
    }

    els.similarModal.classList.remove("hidden");
  }

  // =========================================================================
  // CATALOG LOOKUP DRAWER CONTROLLERS
  // =========================================================================
  async function loadCatalog() {
    if (backendUnavailable) return;
    try {
      catalogData = await api("/products");
      renderCatalogTable(els.catalogSearchInput.value);
    } catch (err) {
      console.warn("Catalog load failed:", err);
      if (err && (err.status === 404 || err.status === 0)) markBackendUnavailable(err);
    }
  }

  function renderCatalogTable(query = "") {
    const q = (query || "").trim().toLowerCase();
    els.catalogTableBody.replaceChildren();

    const filtered = catalogData.filter(
      (item) => !q || `${item.name} ${item.sku} ${item.category || ""} ${item.brand || ""}`.toLowerCase().includes(q)
    );

    if (filtered.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-muted); padding: 20px;">No matching catalog products found.</td>`;
      els.catalogTableBody.appendChild(tr);
      return;
    }

    filtered.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${item.name}</strong></td>
        <td><span class="mono">${item.sku}</span></td>
        <td>${item.category || "—"}</td>
        <td><strong>${rupee(item.price)}</strong></td>
        <td class="text-right">
          <button class="btn btn-primary btn-sm" data-add-sku="${item.id}" type="button">+ Add</button>
        </td>
      `;
      els.catalogTableBody.appendChild(tr);
    });
  }

  // =========================================================================
  // FLORENCE-2 IMAGE TO TEXT ASSISTANT
  // =========================================================================
  async function generateFlorenceCaption() {
    const file = els.captionFileInput.files && els.captionFileInput.files[0];
    if (!file) return;

    els.btnGenerateCaption.disabled = true;
    els.captionResultText.value = "";
    els.captionStatusIndicator.textContent = "Analyzing image with Florence-2...";
    els.captionStatusIndicator.className = "caption-status-msg";

    try {
      const formData = new FormData();
      formData.append("image", file);
      const result = await api("/caption", {
        method: "POST",
        body: formData,
      });
      els.captionResultText.value = result.prompt || "";
      els.captionStatusIndicator.textContent = `✓ Generated (${result.device || "cpu"})`;
      els.captionStatusIndicator.className = "caption-status-msg ok";
    } catch (err) {
      els.captionStatusIndicator.textContent = err.message || "Caption generation failed.";
      els.captionStatusIndicator.className = "caption-status-msg error";
    } finally {
      els.btnGenerateCaption.disabled = !(els.captionFileInput.files && els.captionFileInput.files[0]);
    }
  }

  // =========================================================================
  // EVENT LISTENERS BINDINGS
  // =========================================================================
  function initEventListeners() {
    // 1. Camera Controls
    els.btnCameraToggle.addEventListener("click", toggleCamera);
    els.btnQuickStart.addEventListener("click", toggleCamera);

    els.btnCaptureSnapshot.addEventListener("click", async () => {
      if (!isCameraActive()) return;
      try {
        const blob = await captureCurrentFrameBlob();
        lastCapturedBlob = blob;
        showVisionMessage("Frame snapshot captured.", "info");
      } catch (err) {
        showVisionMessage(err.message, "error");
      }
    });

    // 2. Scan Trigger
    els.btnScanProduct.addEventListener("click", executeProductScan);
    els.btnTryScanAgain.addEventListener("click", resetShowcaseToIdle);
    els.btnScanAgainFound.addEventListener("click", resetShowcaseToIdle);

    // 3. Add to Cart
    els.btnAddDetected.addEventListener("click", addDetectedProductToCart);

    // 4. Cart List Quantity Delegation
    els.cartLinesList.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-act]");
      if (!btn) return;
      const act = btn.dataset.act;
      const id = Number(btn.dataset.id);
      if (id) handleCartQuantityAction(id, act);
    });

    // 5. Clear Cart & New Transaction
    els.btnClearCart.addEventListener("click", async () => {
      if (activeCartItemCount === 0) return;
      if (!confirm("Are you sure you want to clear the active cart?")) return;
      await api("/cart/clear", { method: "POST" });
      await syncCart();
      resetShowcaseToIdle();
      showVisionMessage("Cart cleared.", "info");
    });

    els.btnNewTxn.addEventListener("click", async () => {
      await api("/cart/new", { method: "POST" });
      await syncCart();
      resetShowcaseToIdle();
      showVisionMessage("New transaction started.", "info");
    });

    // 6. Discount Controls
    let discountDebounce = null;
    els.discountInput.addEventListener("input", () => {
      clearTimeout(discountDebounce);
      discountDebounce = setTimeout(() => {
        applyCartDiscount(els.discountInput.value);
      }, 300);
    });

    if (els.discountPresetChips) {
      els.discountPresetChips.forEach((chip) => {
        chip.addEventListener("click", () => {
          const pct = Number(chip.dataset.pct);
          els.discountInput.value = pct;
          applyCartDiscount(pct);
        });
      });
    }

    if (els.btnScanNext) {
      els.btnScanNext.addEventListener("click", () => {
        if (isCameraActive() && !isApiBusy) {
          executeProductScan();
        } else if (!isCameraActive()) {
          toggleCamera();
        }
      });
    }

    // 7. Checkout & Modal Actions
    els.btnGenerateBill.addEventListener("click", executeCheckout);
    els.btnModalNewTxn.addEventListener("click", async () => {
      els.checkoutModal.classList.add("hidden");
      await api("/cart/new", { method: "POST" });
      await syncCart();
      resetShowcaseToIdle();
    });

    // 8. Product Registration Modal
    els.btnRegisterUnknown.addEventListener("click", openProductRegistrationModal);
    els.btnCloseReg.addEventListener("click", closeProductRegistrationModal);
    els.btnCancelReg.addEventListener("click", closeProductRegistrationModal);

    els.regFileInput.addEventListener("change", () => {
      const file = els.regFileInput.files && els.regFileInput.files[0];
      if (file) {
        els.regImgPreview.src = URL.createObjectURL(file);
        els.regImgPreview.style.display = "block";
      }
    });

    els.btnRegRetake.addEventListener("click", async () => {
      if (!isCameraActive()) {
        els.regModalStatus.textContent = "Open camera first to retake photo.";
        return;
      }
      try {
        const blob = await captureCurrentFrameBlob();
        lastCapturedBlob = blob;
        els.regImgPreview.src = URL.createObjectURL(blob);
        els.regImgPreview.style.display = "block";
        els.regModalStatus.textContent = "New photo captured.";
      } catch (err) {
        els.regModalStatus.textContent = err.message || "Capture failed.";
      }
    });

    els.productRegForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        await submitProductRegistration(false);
      } catch (err) {
        els.regModalStatus.textContent = err.message || "Registration failed.";
      }
    });

    // 9. Similar Product Modal Actions
    els.btnUseSimilar.addEventListener("click", async () => {
      els.similarModal.classList.add("hidden");
      if (!currentSimilarProductId) return;
      try {
        await api("/cart/items", {
          method: "POST",
          body: JSON.stringify({ product_id: currentSimilarProductId, quantity: 1 }),
        });
        await syncCart();
        closeProductRegistrationModal();
        resetShowcaseToIdle();
        showVisionMessage("Added existing product to cart.", "success");
      } catch (err) {
        showVisionMessage(err.message, "error");
      }
    });

    els.btnCreateForce.addEventListener("click", async () => {
      els.similarModal.classList.add("hidden");
      try {
        await submitProductRegistration(true);
      } catch (err) {
        els.regModalStatus.textContent = err.message || "Creation failed.";
      }
    });

    // 10. Catalog Drawer
    els.btnOpenCatalog.addEventListener("click", () => {
      loadCatalog();
      els.catalogModal.classList.remove("hidden");
    });
    els.btnCloseCatalog.addEventListener("click", () => {
      els.catalogModal.classList.add("hidden");
    });
    els.catalogSearchInput.addEventListener("input", () => {
      renderCatalogTable(els.catalogSearchInput.value);
    });
    els.catalogTableBody.addEventListener("click", async (e) => {
      const btn = e.target.closest("button[data-add-sku]");
      if (!btn) return;
      const productId = Number(btn.dataset.addSku);
      if (!productId) return;
      try {
        await api("/cart/items", {
          method: "POST",
          body: JSON.stringify({ product_id: productId, quantity: 1 }),
        });
        await syncCart();
        showVisionMessage("Item added from catalog.", "success");
      } catch (err) {
        alert(err.message || "Could not add item.");
      }
    });

    // 11. Florence-2 Caption Modal
    els.btnOpenCaption.addEventListener("click", () => {
      els.captionModal.classList.remove("hidden");
    });
    els.btnCloseCaption.addEventListener("click", () => {
      els.captionModal.classList.add("hidden");
    });
    els.captionFileInput.addEventListener("change", () => {
      const file = els.captionFileInput.files && els.captionFileInput.files[0];
      if (file) {
        els.captionImgPreview.src = URL.createObjectURL(file);
        els.captionImgPreview.classList.remove("hidden");
        els.btnGenerateCaption.disabled = false;
      } else {
        els.captionImgPreview.classList.add("hidden");
        els.btnGenerateCaption.disabled = true;
      }
    });
    els.btnGenerateCaption.addEventListener("click", generateFlorenceCaption);
    els.btnCopyPrompt.addEventListener("click", async () => {
      const text = els.captionResultText.value;
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        els.captionStatusIndicator.textContent = "✓ Prompt copied to clipboard!";
        els.captionStatusIndicator.className = "caption-status-msg ok";
      } catch (_e) {
        els.captionResultText.select();
        document.execCommand("copy");
      }
    });

    // Keyboard Shortcuts (e.g. Spacebar to trigger scan)
    window.addEventListener("keydown", (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      if (e.code === "Space" && isCameraActive() && !isApiBusy) {
        e.preventDefault();
        executeProductScan();
      }
    });

    // Window Resize -> Canvas sync
    window.addEventListener("resize", () => {
      resizeCanvasToDisplaySize();
    });

    window.addEventListener("pagehide", () => {
      userWantsCamera = false;
      stopBrowserCameraStream();
      stopHubPreview();
    });
  }

  // =========================================================================
  // STATUS POLLING & APP BOOTSTRAP
  // =========================================================================
  async function refreshVisionHubStatus() {
    if (backendUnavailable) return;
    try {
      const info = await api("/pos/status");

      // Never reopen or show hub preview after the cashier clicked Close Camera.
      if (!userWantsCamera) {
        if (info.camera_active && !browserCamera.active) {
          stopHubCamera();
        }
        hubVisionStatus = { ...hubVisionStatus, ...info, camera_active: false };
        els.fpsMetaTag.classList.add("hidden");
        els.latencyMetaTag.classList.add("hidden");
        if (!browserCamera.active && currentState !== POS_STATES.CAMERA_OFFLINE && currentState !== POS_STATES.ERROR && currentState !== POS_STATES.CAMERA_STARTING) {
          stopHubPreview();
          els.cameraOfflineOverlay.classList.remove("hidden");
          els.feedLiveBadge.classList.add("hidden");
          setPosState(POS_STATES.CAMERA_OFFLINE);
        }
        return;
      }

      hubVisionStatus = { ...hubVisionStatus, ...info };

      // Update FPS & Latency tags
      if (hubVisionStatus.camera_active) {
        els.fpsValue.textContent = hubVisionStatus.fps || "0";
        els.fpsMetaTag.classList.remove("hidden");
        if (hubVisionStatus.latency_ms) {
          els.latencyValue.textContent = Math.round(hubVisionStatus.latency_ms);
          els.latencyMetaTag.classList.remove("hidden");
        }
      } else {
        els.fpsMetaTag.classList.add("hidden");
        els.latencyMetaTag.classList.add("hidden");
      }

      // If backend camera changed state outside this client
      if (!browserCamera.active) {
        if (hubVisionStatus.camera_active && currentState === POS_STATES.CAMERA_OFFLINE) {
          els.liveFeed.src = `${apiUrl("/pos/stream")}?t=${Date.now()}`;
          els.liveFeed.classList.remove("hidden");
          els.cameraOfflineOverlay.classList.add("hidden");
          els.feedLiveBadge.classList.remove("hidden");
          setPosState(POS_STATES.CAMERA_ONLINE);
        } else if (!hubVisionStatus.camera_active && currentState === POS_STATES.CAMERA_ONLINE) {
          els.liveFeed.classList.add("hidden");
          els.cameraOfflineOverlay.classList.remove("hidden");
          els.feedLiveBadge.classList.add("hidden");
          setPosState(POS_STATES.CAMERA_OFFLINE);
        }
      }
    } catch (_err) {
      if (_err && _err.status === 404) markBackendUnavailable(_err);
    }
  }

  async function boot() {
    initEventListeners();
    if (!backendIsConfigured()) {
      markBackendUnavailable(new Error("API_BASE_URL missing"));
      setPosState(POS_STATES.CAMERA_OFFLINE);
      return;
    }
    await loadCatalog();
    await syncCart();
    await refreshVisionHubStatus();

    setPosState(isCameraActive() && userWantsCamera ? POS_STATES.CAMERA_ONLINE : POS_STATES.CAMERA_OFFLINE);

    // Periodic Background Polls
    pollTimers.push(setInterval(syncCart, 1800));
    pollTimers.push(setInterval(refreshVisionHubStatus, 1800));
  }

  // Run on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
