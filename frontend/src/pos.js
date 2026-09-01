const rupee = (value) => `₹${Number(value ?? 0).toLocaleString("en-IN")}`;

const STATES = {
  CAMERA_CLOSED: "CAMERA_CLOSED",
  CAMERA_OPEN: "CAMERA_OPEN",
  CAPTURING: "CAPTURING",
  SCANNING: "SCANNING",
  PRODUCT_FOUND: "PRODUCT_FOUND",
  PRODUCT_NOT_FOUND: "PRODUCT_NOT_FOUND",
  PRODUCT_AMBIGUOUS: "PRODUCT_AMBIGUOUS",
  MULTIPLE_PRODUCTS: "MULTIPLE_PRODUCTS",
  LOW_IMAGE_QUALITY: "LOW_IMAGE_QUALITY",
  NO_PRODUCT: "NO_PRODUCT",
  ADDING_PRODUCT: "ADDING_PRODUCT",
  PRODUCT_ADDED: "PRODUCT_ADDED",
  ADDED_TO_BILL: "ADDED_TO_BILL",
  ERROR: "ERROR",
};

const els = {
  cameraStatus: document.getElementById("camera-status"),
  txnId: document.getElementById("txn-id"),
  trackMeta: document.getElementById("track-meta"),
  stageHint: document.getElementById("stage-hint"),
  catalogList: document.getElementById("catalog-list"),
  catalogSearch: document.getElementById("catalog-search"),
  cartEmpty: document.getElementById("cart-empty"),
  alerts: document.getElementById("alerts"),
  cartLines: document.getElementById("cart-lines"),
  subtotal: document.getElementById("subtotal"),
  tax: document.getElementById("tax"),
  discount: document.getElementById("discount"),
  grand: document.getElementById("grand"),
  discountInput: document.getElementById("discount-input"),
  btnBill: document.getElementById("btn-bill"),
  btnClear: document.getElementById("btn-clear"),
  btnNew: document.getElementById("btn-new"),
  btnCamera: document.getElementById("btn-camera"),
  btnScan: document.getElementById("btn-scan"),
  scanResult: document.getElementById("scan-result"),
  liveFeed: document.getElementById("live-feed"),
  liveVideo: document.getElementById("live-video"),
  cameraLoading: document.getElementById("camera-loading"),
  productTable: document.getElementById("product-table-body"),
  productModal: document.getElementById("product-modal"),
  productForm: document.getElementById("product-form"),
  productPreview: document.getElementById("product-preview"),
  productFile: document.getElementById("product-file"),
  productCancel: document.getElementById("product-cancel"),
  productStatus: document.getElementById("product-modal-status"),
  btnRetake: document.getElementById("btn-retake"),
  camIndicator: document.getElementById("cam-indicator"),
  scanIndicator: document.getElementById("scan-indicator"),
  visionMessage: document.getElementById("vision-message"),
  modal: document.getElementById("modal"),
  modalInvoice: document.getElementById("modal-invoice"),
  modalTotal: document.getElementById("modal-total"),
  modalPdf: document.getElementById("modal-pdf"),
  modalClose: document.getElementById("modal-close"),
  successModal: document.getElementById("success-modal"),
  successName: document.getElementById("success-name"),
  successPrice: document.getElementById("success-price"),
  successScan: document.getElementById("success-scan"),
  successContinue: document.getElementById("success-continue"),
  btnScanNext: document.getElementById("btn-scan-next"),
  similarModal: document.getElementById("similar-modal"),
  similarPreview: document.getElementById("similar-preview"),
  similarName: document.getElementById("similar-name"),
  similarPrice: document.getElementById("similar-price"),
  similarUse: document.getElementById("similar-use"),
  similarCreate: document.getElementById("similar-create"),
  captionFile: document.getElementById("caption-file"),
  captionPreview: document.getElementById("caption-preview"),
  btnCaption: document.getElementById("btn-caption"),
  captionText: document.getElementById("caption-text"),
  btnCopyCaption: document.getElementById("btn-copy-caption"),
  captionStatus: document.getElementById("caption-status"),
};

let catalog = [];
let vision = { camera_active: false, detection_active: false, detection_loading: false };
let uiState = STATES.CAMERA_CLOSED;
let lastScan = null;
let registerTrackId = null;
let registerFile = null;
let registerScanId = null;
let lastRegisteredProduct = null;
let lastItemCount = 0;
let pendingRegisterForm = null;
let similarProductId = null;
const browserCamera = { active: false, stream: null };

function cameraIsOn() {
  return Boolean(browserCamera.active || vision.camera_active);
}

function setCameraLoading(on, text = "Opening camera...") {
  if (!els.cameraLoading) return;
  els.cameraLoading.textContent = text;
  els.cameraLoading.classList.toggle("hidden", !on);
}

function stopBrowserCamera() {
  if (browserCamera.stream) {
    browserCamera.stream.getTracks().forEach((track) => track.stop());
  }
  browserCamera.stream = null;
  browserCamera.active = false;
  if (els.liveVideo) {
    els.liveVideo.srcObject = null;
    els.liveVideo.classList.add("hidden");
  }
  if (els.liveFeed) els.liveFeed.classList.remove("hidden");
}

function cameraErrorMessage(error) {
  const name = error && error.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera permission denied. Please allow camera access in your browser.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "Camera unavailable. Please check your webcam connection.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Camera initialization failed. It may already be in use.";
  }
  return (error && error.message) || "Camera initialization failed.";
}

function captureVideoFrame(video, maxSide = 1280, quality = 0.92) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) {
    return Promise.reject(new Error("Camera initialization failed."));
  }
  let width = vw;
  let height = vh;
  if (Math.max(width, height) > maxSide) {
    const scale = maxSide / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(video, 0, 0, width, height);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error("Image processing failed."));
      else resolve(blob);
    }, "image/jpeg", quality);
  });
}

function setState(next, message = "") {
  uiState = next;
  if (message) showVisionMessage(message);
  const busy = [STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState);
  const cameraOn = cameraIsOn();
  els.btnCamera.disabled = busy;
  els.btnScan.disabled = busy || !cameraOn;
  els.btnBill.disabled = busy || lastItemCount === 0;
  els.scanIndicator.textContent = {
    [STATES.CAMERA_CLOSED]: "Status: Camera closed",
    [STATES.CAMERA_OPEN]: "Status: Camera open — ready to scan",
    [STATES.CAPTURING]: "Status: Capturing frame...",
    [STATES.SCANNING]: "Status: Scanning product...",
    [STATES.PRODUCT_FOUND]: "Status: Product found",
    [STATES.PRODUCT_NOT_FOUND]: "Status: Product not found",
    [STATES.PRODUCT_AMBIGUOUS]: "Status: Ambiguous match — scan again",
    [STATES.MULTIPLE_PRODUCTS]: "Status: Multiple products detected",
    [STATES.LOW_IMAGE_QUALITY]: "Status: Image quality too low",
    [STATES.NO_PRODUCT]: "Status: No product detected",
    [STATES.ADDING_PRODUCT]: "Status: Adding product...",
    [STATES.PRODUCT_ADDED]: "Status: Product added",
    [STATES.ADDED_TO_BILL]: "Status: Added to bill",
    [STATES.ERROR]: "Status: Error",
  }[uiState] || "Status: Ready";
  els.scanIndicator.classList.toggle("detecting", busy || uiState === STATES.PRODUCT_FOUND);
}

function showVisionMessage(text) {
  const message = (text || "").trim();
  els.visionMessage.classList.toggle("hidden", !message);
  els.visionMessage.textContent = message;
}

function applyVisionState(info) {
  vision = {
    camera_active: Boolean(info.camera_active),
    detection_active: Boolean(info.detection_active),
    detection_loading: Boolean(info.detection_loading),
  };
  if (browserCamera.active) {
    setServerPreview(false);
    els.btnCamera.textContent = "Close Camera";
    els.camIndicator.textContent = "● Camera Active";
    els.camIndicator.classList.add("active");
    els.camIndicator.classList.remove("offline");
    els.cameraStatus.textContent = "camera active";
    els.cameraStatus.classList.add("live");
    els.cameraStatus.classList.remove("warn");
    els.trackMeta.textContent = "browser camera · preview";
    els.stageHint.textContent = "Place a product in the frame, then click Scan Product.";
    const hold = [
      STATES.SCANNING,
      STATES.CAPTURING,
      STATES.PRODUCT_FOUND,
      STATES.PRODUCT_NOT_FOUND,
      STATES.PRODUCT_AMBIGUOUS,
      STATES.ADDING_PRODUCT,
      STATES.PRODUCT_ADDED,
      STATES.ADDED_TO_BILL,
    ];
    if (!hold.includes(uiState)) {
      setState(STATES.CAMERA_OPEN);
    } else {
      const busy = [STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState);
      els.btnScan.disabled = busy;
      els.btnBill.disabled = busy || lastItemCount === 0;
    }
    return;
  }
  const cameraOn = vision.camera_active;
  setServerPreview(cameraOn);
  els.btnCamera.textContent = cameraOn ? "Close Camera" : "Open Camera";
  els.camIndicator.textContent = cameraOn ? "● Camera Active" : "● Camera Offline";
  els.camIndicator.classList.toggle("active", cameraOn);
  els.camIndicator.classList.toggle("offline", !cameraOn);
  els.cameraStatus.textContent = cameraOn ? "camera active" : "camera offline";
  els.cameraStatus.classList.toggle("live", cameraOn);
  els.cameraStatus.classList.toggle("warn", Boolean(info.error));
  els.trackMeta.textContent = cameraOn ? `${info.fps || 0} fps · preview` : "camera off";
  els.stageHint.textContent = cameraOn
    ? "Place a product in the frame, then click Scan Product."
    : "Open the camera, place one product in view, then click Scan Product.";

  if (info.error) {
    setState(STATES.ERROR, info.error);
  } else if (cameraOn && ![STATES.SCANNING, STATES.CAPTURING, STATES.PRODUCT_FOUND, STATES.PRODUCT_NOT_FOUND, STATES.ADDING_PRODUCT, STATES.PRODUCT_ADDED, STATES.ADDED_TO_BILL].includes(uiState)) {
    setState(STATES.CAMERA_OPEN);
  } else if (!cameraOn && uiState !== STATES.PRODUCT_ADDED) {
    setState(STATES.CAMERA_CLOSED);
  } else {
    els.btnScan.disabled = [STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState) || !cameraOn;
    els.btnBill.disabled = [STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState) || lastItemCount === 0;
  }
}

function mediaUrl(path) {
  return window.RetailVisionAPI ? window.RetailVisionAPI.mediaUrl(path) : path;
}

function setServerPreview(on) {
  if (!els.liveFeed) return;
  if (on) {
    els.liveFeed.src = `${mediaUrl("/pos/stream")}?t=${Date.now()}`;
    els.liveFeed.classList.remove("hidden");
  } else {
    els.liveFeed.removeAttribute("src");
  }
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers["Content-Type"] && options.body) {
    headers["Content-Type"] = "application/json";
  }
  const url = window.RetailVisionAPI ? window.RetailVisionAPI.apiUrl(path) : path;
  const response = await fetch(url, { headers, ...options });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_error) {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

function renderCatalog(filter = "") {
  const query = filter.trim().toLowerCase();
  els.catalogList.innerHTML = "";
  catalog
    .filter((item) => !query || `${item.sku} ${item.name}`.toLowerCase().includes(query))
    .forEach((item) => {
      const button = document.createElement("button");
      button.className = "sku-chip";
      button.type = "button";
      button.textContent = `${item.name} · ₹${item.price}`;
      button.addEventListener("click", async () => {
        await api("/cart/items", {
          method: "POST",
          body: JSON.stringify({ product_id: item.id, quantity: 1 }),
        });
        await refreshCart();
        setState(STATES.ADDED_TO_BILL, `${item.name} added to bill.`);
      });
      els.catalogList.appendChild(button);
    });
  renderProductTable();
}

function renderProductTable() {
  if (!els.productTable) return;
  els.productTable.innerHTML = "";
  catalog.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.name}</td><td>${item.sku}</td><td>${rupee(item.price)}</td><td></td>`;
    const actions = row.querySelector("td:last-child");
    const edit = document.createElement("button");
    edit.type = "button";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => openEditProduct(item));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => removeProduct(item));
    actions.append(edit, remove);
    els.productTable.appendChild(row);
  });
}

function clearScanResult() {
  els.scanResult.classList.add("hidden");
  els.scanResult.replaceChildren();
  lastScan = null;
}

function renderFoundCard(product) {
  els.scanResult.classList.remove("hidden");
  els.scanResult.replaceChildren();
  const card = document.createElement("div");
  card.className = "result-card found";
  const img = product.image_url
    ? `<img src="${mediaUrl(product.image_url)}?t=${Date.now()}" alt="${product.name}" onerror="this.style.display='none'" />`
    : "";
  const confidence = Math.round(Number(product.confidence || 0) * 100);
  card.innerHTML = `
    <strong>PRODUCT FOUND ✓</strong>
    ${img}
    <h3>${product.name}</h3>
    <p>Price: ${rupee(product.price)}</p>
    <p>Confidence: ${confidence}%</p>
  `;
  const add = document.createElement("button");
  add.type = "button";
  add.className = "vision-btn detect";
  add.textContent = "ADD TO BILL";
  add.addEventListener("click", () => addFoundToBill(product));
  const again = document.createElement("button");
  again.type = "button";
  again.className = "vision-btn";
  again.textContent = "SCAN AGAIN";
  again.addEventListener("click", readyToScanAgain);
  card.append(add, again);
  els.scanResult.appendChild(card);
}

function readyToScanAgain() {
  clearScanResult();
  setState(cameraIsOn() ? STATES.CAMERA_OPEN : STATES.CAMERA_CLOSED, "Ready to scan.");
}

function renderActionCard({ title, message, previewUrl, state, allowAddNew = false, className = "missing" }) {
  els.scanResult.classList.remove("hidden");
  els.scanResult.replaceChildren();
  const card = document.createElement("div");
  card.className = `result-card ${className}`;
  const preview = previewUrl
    ? `<img src="${mediaUrl(previewUrl)}?t=${Date.now()}" alt="Captured product" onerror="this.style.display='none'" />`
    : "";
  card.innerHTML = `
    <strong>${title}</strong>
    ${preview}
    <p>${(message || "").replace(/\n/g, "<br/>")}</p>
  `;
  const again = document.createElement("button");
  again.type = "button";
  again.className = "vision-btn detect";
  again.textContent = "SCAN AGAIN";
  again.addEventListener("click", readyToScanAgain);
  card.append(again);
  if (allowAddNew) {
    const add = document.createElement("button");
    add.type = "button";
    add.className = "vision-btn";
    add.textContent = "ADD NEW PRODUCT";
    add.addEventListener("click", () => openRegisterFromScan(lastScan || {}));
    card.append(add);
  }
  els.scanResult.appendChild(card);
  if (state) setState(state, message || title);
}

function renderNotFoundCard(payload) {
  const status = payload.status || "";
  if (status === "multiple_products") {
    renderActionCard({
      title: "MULTIPLE PRODUCTS DETECTED",
      message: payload.message || "Please place only one product inside the scanning area.",
      previewUrl: payload.preview_url,
      state: STATES.MULTIPLE_PRODUCTS,
      allowAddNew: false,
      className: "warn",
    });
    return;
  }
  if (status === "low_image_quality") {
    renderActionCard({
      title: "IMAGE QUALITY TOO LOW",
      message: payload.message || "Please hold the product steady and scan again.",
      previewUrl: payload.preview_url,
      state: STATES.LOW_IMAGE_QUALITY,
      allowAddNew: false,
      className: "warn",
    });
    return;
  }
  if (status === "no_product") {
    renderActionCard({
      title: "NO PRODUCT DETECTED",
      message: payload.message || "Please place only one product inside the scanning area.",
      previewUrl: payload.preview_url,
      state: STATES.NO_PRODUCT,
      allowAddNew: true,
      className: "warn",
    });
    return;
  }
  const ambiguous = status === "ambiguous" || payload.reason === "ambiguous_match";
  if (ambiguous) {
    renderActionCard({
      title: "AMBIGUOUS MATCH",
      message: payload.message || "Multiple visually similar products found. Please scan again.",
      previewUrl: payload.preview_url,
      state: STATES.PRODUCT_AMBIGUOUS,
      allowAddNew: false,
      className: "warn",
    });
    return;
  }
  const unknown = status === "unknown" || status === "not_found";
  if (unknown) {
    renderActionCard({
      title: "Unknown Product",
      message: payload.message || "We couldn't confidently identify this product.",
      previewUrl: payload.preview_url,
      state: STATES.PRODUCT_NOT_FOUND,
      allowAddNew: true,
      className: "missing",
    });
    return;
  }
  renderActionCard({
    title: "PRODUCT NOT FOUND",
    message: payload.message || "We couldn't confidently identify this product.",
    previewUrl: payload.preview_url,
    state: STATES.PRODUCT_NOT_FOUND,
    allowAddNew: true,
    className: "missing",
  });
}

async function addFoundToBill(product) {
  try {
    await api("/cart/items", {
      method: "POST",
      body: JSON.stringify({ product_id: product.product_id, quantity: 1 }),
    });
    await refreshCart();
    setState(STATES.ADDED_TO_BILL, `${product.name} added to bill.`);
    clearScanResult();
  } catch (error) {
    setState(STATES.ERROR, error.message || "Could not add product to bill.");
  }
}

function openRegisterFromScan(payload) {
  registerTrackId = null;
  registerFile = null;
  registerScanId = payload.scan_id || null;
  delete els.productForm.dataset.editId;
  els.productForm.reset();
  document.getElementById("f-tax").value = "18";
  els.productStatus.textContent = "";
  if (payload.preview_url) {
    els.productPreview.src = `${mediaUrl(payload.preview_url)}?t=${Date.now()}`;
  } else {
    els.productPreview.removeAttribute("src");
  }
  setState(STATES.ADDING_PRODUCT);
  els.productModal.classList.remove("hidden");
}

function openRegisterModal(trackId, cropUrl) {
  registerTrackId = trackId ?? null;
  registerFile = null;
  registerScanId = null;
  delete els.productForm.dataset.editId;
  els.productForm.reset();
  document.getElementById("f-tax").value = "18";
  els.productStatus.textContent = "";
  if (cropUrl) {
    els.productPreview.src = `${cropUrl}?t=${Date.now()}`;
  } else {
    els.productPreview.removeAttribute("src");
  }
  els.productModal.classList.remove("hidden");
}

function openEditProduct(item) {
  registerTrackId = null;
  registerFile = null;
  registerScanId = null;
  els.productForm.reset();
  document.getElementById("f-name").value = item.name;
  const skuField = document.getElementById("f-sku");
  if (skuField) skuField.value = item.sku || "";
  document.getElementById("f-brand").value = item.brand || "";
  document.getElementById("f-category").value = item.category || "";
  const variantField = document.getElementById("f-variant");
  if (variantField) variantField.value = item.variant || "";
  const weightField = document.getElementById("f-weight");
  if (weightField) weightField.value = item.weight || "";
  document.getElementById("f-price").value = item.price;
  document.getElementById("f-tax").value = item.tax_rate ?? 18;
  els.productPreview.removeAttribute("src");
  els.productStatus.textContent = "Editing existing product. Save updates the catalog.";
  els.productForm.dataset.editId = String(item.id);
  els.productModal.classList.remove("hidden");
}

async function removeProduct(item) {
  if (!confirm(`Soft-delete ${item.name}? Historical bills keep the original name.`)) return;
  await api(`/products/${item.id}`, { method: "DELETE" });
  catalog = await api("/products");
  renderCatalog(els.catalogSearch.value);
}

els.productCancel.addEventListener("click", () => {
  els.productModal.classList.add("hidden");
  delete els.productForm.dataset.editId;
  setState(vision.camera_active ? STATES.CAMERA_OPEN : STATES.CAMERA_CLOSED);
});

els.btnRetake.addEventListener("click", async () => {
  if (browserCamera.active) {
    try {
      const blob = await captureVideoFrame(els.liveVideo);
      registerFile = new File([blob], "retake.jpg", { type: "image/jpeg" });
      registerScanId = null;
      els.productPreview.src = URL.createObjectURL(blob);
      els.productStatus.textContent = "New image captured.";
    } catch (error) {
      els.productStatus.textContent = error.message || "Retake failed.";
    }
    return;
  }
  if (!vision.camera_active) {
    els.productStatus.textContent = "Open the camera first to retake an image.";
    return;
  }
  try {
    const response = await fetch(mediaUrl("/pos/camera/capture"), { method: "POST" });
    if (!response.ok) throw new Error("Could not capture a new image.");
    const blob = await response.blob();
    registerFile = new File([blob], "retake.jpg", { type: "image/jpeg" });
    registerScanId = null;
    els.productPreview.src = URL.createObjectURL(blob);
    els.productStatus.textContent = "New image captured.";
  } catch (error) {
    els.productStatus.textContent = error.message || "Retake failed.";
  }
});

els.productFile.addEventListener("change", () => {
  registerFile = els.productFile.files[0] || null;
  if (registerFile) {
    registerScanId = null;
    els.productPreview.src = URL.createObjectURL(registerFile);
  }
});

els.productForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.productStatus.textContent = "";
  const editId = els.productForm.dataset.editId;
  try {
    setState(STATES.ADDING_PRODUCT, "Adding product...");
    if (editId) {
      await api(`/products/${editId}`, {
        method: "PUT",
        body: JSON.stringify({
          name: document.getElementById("f-name").value,
          sku: (document.getElementById("f-sku").value || "").trim() || undefined,
          brand: document.getElementById("f-brand").value || null,
          category: document.getElementById("f-category").value || null,
          variant: (document.getElementById("f-variant") || {}).value || null,
          weight: (document.getElementById("f-weight") || {}).value || null,
          price: Number(document.getElementById("f-price").value),
          tax_rate: Number(document.getElementById("f-tax").value || 18),
        }),
      });
      els.productModal.classList.add("hidden");
    } else {
      await submitRegisterProduct(false);
    }
    catalog = await api("/products");
    renderCatalog(els.catalogSearch.value);
    await refreshCart();
    delete els.productForm.dataset.editId;
  } catch (error) {
    els.productStatus.textContent = error.message || "Product could not be saved. Please try again.";
    setState(STATES.ERROR, error.message || "Product creation failed.");
  }
});

async function submitRegisterProduct(forceCreate) {
  const data = new FormData();
  data.append("name", document.getElementById("f-name").value);
  const sku = (document.getElementById("f-sku").value || "").trim();
  if (sku) data.append("sku", sku);
  data.append("price", document.getElementById("f-price").value);
  data.append("tax_rate", document.getElementById("f-tax").value || "18");
  data.append("brand", document.getElementById("f-brand").value);
  data.append("category", document.getElementById("f-category").value);
  data.append("variant", (document.getElementById("f-variant") || {}).value || "");
  data.append("weight", (document.getElementById("f-weight") || {}).value || "");
  data.append("add_to_cart", "true");
  data.append("force_create", forceCreate ? "true" : "false");
  if (registerTrackId != null) data.append("track_id", String(registerTrackId));
  if (registerScanId) data.append("scan_id", registerScanId);
  if (registerFile) data.append("image", registerFile);

  const response = await fetch(mediaUrl("/products/register"), { method: "POST", body: data });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 409 && payload.detail && payload.detail.status === "similar_product_found") {
    showSimilarProduct(payload.detail);
    throw new Error(payload.detail.message || "Similar product found.");
  }
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(typeof detail === "string" ? detail : (detail && detail.message) || "Product could not be saved.");
  }
  lastRegisteredProduct = payload;
  els.productModal.classList.add("hidden");
  clearScanResult();
  els.successName.textContent = `Product: ${payload.name}`;
  els.successPrice.textContent = `Price: ${rupee(payload.price)}`;
  els.successModal.classList.remove("hidden");
  setState(STATES.PRODUCT_ADDED, payload.message || "Product added successfully.");
  return payload;
}

function showSimilarProduct(detail) {
  const product = detail.product || {};
  similarProductId = product.id;
  pendingRegisterForm = true;
  if (product.image_url) {
    els.similarPreview.src = `${mediaUrl(product.image_url)}?t=${Date.now()}`;
    els.similarPreview.style.display = "block";
  } else {
    els.similarPreview.removeAttribute("src");
    els.similarPreview.style.display = "none";
  }
  els.similarName.textContent = product.name || "Existing product";
  els.similarPrice.textContent = `Price: ${rupee(product.price)}`;
  els.similarModal.classList.remove("hidden");
}

if (els.similarUse) {
  els.similarUse.addEventListener("click", async () => {
    els.similarModal.classList.add("hidden");
    els.productModal.classList.add("hidden");
    if (!similarProductId) return;
    try {
      await api("/cart/items", {
        method: "POST",
        body: JSON.stringify({ product_id: similarProductId, quantity: 1 }),
      });
      await refreshCart();
      setState(STATES.ADDED_TO_BILL, "Existing product added to bill.");
      clearScanResult();
    } catch (error) {
      setState(STATES.ERROR, error.message || "Could not add product.");
    }
  });
}

if (els.similarCreate) {
  els.similarCreate.addEventListener("click", async () => {
    els.similarModal.classList.add("hidden");
    try {
      setState(STATES.ADDING_PRODUCT, "Creating new product...");
      await submitRegisterProduct(true);
      catalog = await api("/products");
      renderCatalog(els.catalogSearch.value);
      await refreshCart();
    } catch (error) {
      els.productStatus.textContent = error.message || "Product could not be saved.";
      setState(STATES.ERROR, error.message || "Product creation failed.");
    }
  });
}

els.successContinue.addEventListener("click", () => {
  els.successModal.classList.add("hidden");
  setState(vision.camera_active ? STATES.CAMERA_OPEN : STATES.CAMERA_CLOSED, "Ready to scan.");
});

els.successScan.addEventListener("click", async () => {
  els.successModal.classList.add("hidden");
  if (!cameraIsOn()) {
    showVisionMessage("Open the camera, then scan the newly added product.");
    setState(STATES.CAMERA_CLOSED);
    return;
  }
  await runScan();
});

function renderCart(cart) {
  lastItemCount = (cart.items || []).length;
  els.txnId.textContent = `TXN ${cart.transaction_id}`;
  els.subtotal.textContent = rupee(cart.subtotal);
  els.tax.textContent = rupee(cart.tax);
  els.discount.textContent = rupee(cart.discount);
  els.grand.textContent = rupee(cart.grand_total);
  if (document.activeElement !== els.discountInput) {
    els.discountInput.value = cart.discount_percent ?? 0;
  }
  els.cartLines.innerHTML = "";
  els.cartEmpty.style.display = cart.items.length ? "none" : "block";
  const busy = [STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState);
  els.btnBill.disabled = busy || cart.items.length === 0;

  const alerts = cart.alerts || [];
  els.alerts.classList.toggle("hidden", alerts.length === 0);
  els.alerts.replaceChildren();
  alerts.forEach((item) => {
    const pre = document.createElement("pre");
    pre.textContent = item.message || "UNKNOWN PRODUCT\nPlease verify manually.";
    els.alerts.appendChild(pre);
  });

  cart.items.forEach((item) => {
    const line = document.createElement("li");
    line.className = "cart-line";
    line.innerHTML = `
      <div>
        <h3>${item.name}</h3>
        <p>${item.quantity} × ${rupee(item.unit_price)}</p>
        <div class="controls">
          <button data-act="dec" data-id="${item.product_id}" type="button">−</button>
          <button type="button" disabled>×${item.quantity}</button>
          <button data-act="inc" data-id="${item.product_id}" type="button">+</button>
          <button class="danger" data-act="remove" data-id="${item.product_id}" type="button">Remove</button>
        </div>
      </div>
      <div class="line-total">${rupee(item.total)}</div>
    `;
    els.cartLines.appendChild(line);
  });
}

async function refreshCart() {
  const cart = await api("/cart");
  renderCart(cart);
}

async function refreshStatus() {
  try {
    const info = await api("/pos/status");
    applyVisionState(info);
  } catch (_error) {
    els.cameraStatus.textContent = "offline";
  }
}

async function runScan() {
  if (!cameraIsOn()) {
    setState(STATES.ERROR, "Please turn on the camera first.");
    return;
  }
  clearScanResult();
  setCameraLoading(true, "Scanning product...");
  setState(STATES.CAPTURING, "Capturing frame...");
  setState(STATES.SCANNING, "Scanning product...");
  showVisionMessage("Searching product database...");
  try {
    const data = new FormData();
    if (browserCamera.active) {
      const blob = await captureVideoFrame(els.liveVideo);
      data.append("use_camera", "false");
      data.append("image", blob, "scan.jpg");
    } else {
      data.append("use_camera", "true");
    }
    const result = await api("/products/scan", { method: "POST", body: data });
    lastScan = result;
    if ((result.found || result.status === "found") && result.product) {
      setState(STATES.PRODUCT_FOUND, "Product found!");
      renderFoundCard(result.product);
    } else {
      renderNotFoundCard(result);
    }
  } catch (error) {
    setState(STATES.ERROR, error.message || "Recognition service unavailable.");
    clearScanResult();
  } finally {
    setCameraLoading(false);
  }
}

els.cartLines.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-act]");
  if (!button) return;
  const id = button.dataset.id;
  const act = button.dataset.act;
  const routes = {
    inc: ["/cart/items/" + id + "/increase", "POST"],
    dec: ["/cart/items/" + id + "/decrease", "POST"],
    remove: ["/cart/items/" + id, "DELETE"],
  };
  const [path, method] = routes[act];
  await api(path, { method });
  await refreshCart();
});

els.btnClear.addEventListener("click", async () => {
  await api("/cart/clear", { method: "POST" });
  await refreshCart();
});

els.btnNew.addEventListener("click", async () => {
  await api("/cart/new", { method: "POST" });
  clearScanResult();
  await refreshCart();
});

els.btnBill.addEventListener("click", async () => {
  if ([STATES.CAPTURING, STATES.SCANNING, STATES.ADDING_PRODUCT].includes(uiState)) return;
  els.btnBill.disabled = true;
  try {
    const result = await api("/checkout", { method: "POST" });
    els.modalInvoice.textContent = result.invoice_number;
    els.modalTotal.textContent = `Grand total ${rupee(result.bill.grand_total)}`;
    els.modalPdf.href = mediaUrl(result.pdf_url);
    els.modal.classList.remove("hidden");
    await refreshCart();
  } catch (error) {
    alert(error.message);
  } finally {
    els.btnBill.disabled = false;
  }
});

els.modalClose.addEventListener("click", () => {
  els.modal.classList.add("hidden");
});

let discountTimer = 0;
els.discountInput.addEventListener("change", async () => {
  clearTimeout(discountTimer);
  discountTimer = setTimeout(async () => {
    await api("/cart/discount", {
      method: "POST",
      body: JSON.stringify({ percent: Number(els.discountInput.value || 0) }),
    });
    await refreshCart();
  }, 200);
});

els.catalogSearch.addEventListener("input", () => renderCatalog(els.catalogSearch.value));

function setCaptionStatus(text, kind = "") {
  if (!els.captionStatus) return;
  els.captionStatus.textContent = text || "";
  els.captionStatus.classList.toggle("error", kind === "error");
  els.captionStatus.classList.toggle("ok", kind === "ok");
}

if (els.captionFile) {
  els.captionFile.addEventListener("change", () => {
    const file = els.captionFile.files && els.captionFile.files[0];
    if (!file) {
      els.btnCaption.disabled = true;
      els.captionPreview.classList.add("hidden");
      els.captionPreview.removeAttribute("src");
      setCaptionStatus("");
      return;
    }
    els.captionPreview.src = URL.createObjectURL(file);
    els.captionPreview.classList.remove("hidden");
    els.btnCaption.disabled = false;
    els.captionText.value = "";
    setCaptionStatus("Image ready. Click Generate Prompt.");
  });
}

if (els.btnCaption) {
  els.btnCaption.addEventListener("click", async () => {
    const file = els.captionFile && els.captionFile.files && els.captionFile.files[0];
    if (!file) {
      setCaptionStatus("Upload an image first.", "error");
      return;
    }
    els.btnCaption.disabled = true;
    els.captionText.value = "";
    setCaptionStatus("Generating prompt… first run downloads Florence-2 and can take a few minutes.");
    try {
      const data = new FormData();
      data.append("image", file);
      const result = await api("/caption", { method: "POST", body: data });
      els.captionText.value = result.prompt || "";
      setCaptionStatus(`Done (${result.device || "cpu"} · ${result.model || "Florence-2"}).`, "ok");
    } catch (error) {
      setCaptionStatus(error.message || "Caption generation failed.", "error");
    } finally {
      els.btnCaption.disabled = !(els.captionFile && els.captionFile.files && els.captionFile.files[0]);
    }
  });
}

if (els.btnCopyCaption) {
  els.btnCopyCaption.addEventListener("click", async () => {
    const text = (els.captionText && els.captionText.value) || "";
    if (!text) {
      setCaptionStatus("Nothing to copy yet.", "error");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCaptionStatus("Prompt copied.", "ok");
    } catch (_error) {
      els.captionText.select();
      document.execCommand("copy");
      setCaptionStatus("Prompt copied.", "ok");
    }
  });
}

els.btnCamera.addEventListener("click", async () => {
  els.btnCamera.disabled = true;
  try {
    if (cameraIsOn()) {
      setCameraLoading(true, "Closing camera...");
      stopBrowserCamera();
      if (vision.camera_active) {
        const info = await api("/pos/camera/stop", { method: "POST" });
        applyVisionState(info);
      } else {
        vision.camera_active = false;
        applyVisionState({ camera_active: false, detection_active: false, detection_loading: false });
      }
      setCameraLoading(false);
      setState(STATES.CAMERA_CLOSED);
      clearScanResult();
      return;
    }
    setCameraLoading(true, "Opening camera...");
    showVisionMessage("Opening camera...");
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
          audio: false,
        });
        browserCamera.stream = stream;
        browserCamera.active = true;
        els.liveVideo.srcObject = stream;
        els.liveVideo.classList.remove("hidden");
        els.liveFeed.classList.add("hidden");
        setCameraLoading(false);
        applyVisionState({ camera_active: false, detection_active: false, detection_loading: false });
        setState(STATES.CAMERA_OPEN, "Camera ready. Click Scan Product when the item is in view.");
        return;
      } catch (browserError) {
        const friendly = cameraErrorMessage(browserError);
        if (browserError.name === "NotAllowedError" || browserError.name === "PermissionDeniedError") {
          setCameraLoading(false);
          setState(STATES.ERROR, friendly);
          return;
        }
        showVisionMessage(`${friendly} Trying the checkout camera...`);
      }
    }
    const info = await api("/pos/camera/start", { method: "POST" });
    setCameraLoading(false);
    applyVisionState(info);
    if (!info.camera_active && info.error) {
      setState(STATES.ERROR, info.error);
    } else if (info.camera_active) {
      setState(STATES.CAMERA_OPEN, "Camera ready. Click Scan Product when the item is in view.");
    } else {
      setState(STATES.CAMERA_CLOSED);
      clearScanResult();
    }
  } catch (error) {
    setCameraLoading(false);
    stopBrowserCamera();
    setState(
      STATES.ERROR,
      cameraErrorMessage(error) || "Camera could not be started.\nPlease check your webcam connection.",
    );
  } finally {
    els.btnCamera.disabled = false;
    setCameraLoading(false);
  }
});

els.btnScan.addEventListener("click", () => {
  runScan();
});

if (els.btnScanNext) {
  els.btnScanNext.addEventListener("click", () => {
    readyToScanAgain();
    if (!cameraIsOn()) {
      showVisionMessage("Open the camera, then scan the next product.");
    }
  });
}

async function boot() {
  catalog = await api("/products");
  renderCatalog();
  await refreshCart();
  await refreshStatus();
  setState(STATES.CAMERA_CLOSED);
  setInterval(refreshCart, 1200);
  setInterval(refreshStatus, 1200);
}

boot().catch((error) => {
  els.cameraStatus.textContent = "POS error";
  els.cartEmpty.textContent = error.message;
});
