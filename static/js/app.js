const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const home = $("#home");
const workspaceView = $("#workspaceView");
const backHome = $("#backHome");
const dropZone = $("#dropZone");
const sidebarDrop = $("#sidebarDrop");
const sidebar = $(".sidebar");
const fileInput = $("#fileInput");
const workspace = $("#workspace");
const imageList = $("#imageList");
const mainImage = $("#mainImage");
const overlay = $("#overlay");
const canvasWrap = $("#canvasWrap");
const loading = $("#loading");
const results = $("#results");
const clickControls = $("#clickControls");
const upscaleControls = $("#upscaleControls");
const chromaControls = $("#chromaControls");
const chromaKeyToggle = $("#chromaKeyToggle");
const chromaColor = $("#chromaColor");
const borderThickness = $("#borderThickness");
const chromaTolerance = $("#chromaTolerance");
const hint = $("#hint");
const animatorPanel = $("#animatorPanel");
const animationPreview = $("#animationPreview");
const animationVideo = $("#animationVideo");
const animationEmpty = $("#animationEmpty");
const animationFrames = $("#animationFrames");
const animationLibrary = $("#animationLibrary");
const clearAnimationFrames = $("#clearAnimationFrames");
const animationInterval = $("#animationInterval");
const animationIntervalValue = $("#animationIntervalValue");
const animationFormat = $("#animationFormat");
const animationSmooth = $("#animationSmooth");
const animationResult = $("#animationResult");
const buildAnimationButton = $("#buildAnimation");
const runCurrentTask = $("#runCurrentTask");
const objectPanel = $("#objectPanel");
const historyPanel = $("#historyPanel");
const historyList = $("#historyList");
const queueMenu = $("#queueMenu");
const cropDialog = $("#cropDialog");
const cropStage = $("#cropStage");
const cropImage = $("#cropImage");
const cropSelection = $("#cropSelection");

let images = [];
let current = null;
let mode = "bg";
let pointMode = "foreground";
let objects = [];        // click-mode selections: [{ points:[{x,y,label}], maskUrl, bbox, tinted }]
let activeIndex = -1;
let processedTasks = new Map();
let autoRunSettings = new Map();
let segmentationInProgress = false;
let menuImage = null;
let cropImageSource = null;
let cropStart = null;
let animationSources = [];
let animationLibrarySources = [];
let animationTimer = null;
let autoExtractFiles = [];

const animatorHistoryModes = new Set(["bg", "click", "auto"]);
const animatorImageExtensions = /\.(png|jpe?g|webp|bmp)$/i;

function syncChromaValues() {
  $("#borderThicknessValue").textContent = `${Number(borderThickness.value).toFixed(2)} pt`;
  $("#chromaToleranceValue").textContent = chromaTolerance.value;
}

[borderThickness, chromaTolerance].forEach(input => input.addEventListener("input", syncChromaValues));
syncChromaValues();

function taskKey(img) {
  return `${img?.job_id || "single"}:${img?.image_id || ""}`;
}

function shortId(value) {
  const text = String(value ?? "");
  return text.length > 10 ? text.slice(0, 10) : text;
}

document.addEventListener("click", (ev) => {
  if (!ev.target.closest("#queueMenu")) queueMenu.classList.add("hidden");
});

queueMenu.addEventListener("click", async (ev) => {
  const action = ev.target.closest("button")?.dataset.action;
  if (!action || !menuImage) return;
  const img = menuImage;
  queueMenu.classList.add("hidden");
  if (action === "duplicate") await duplicateImage(img);
  if (action === "crop") openCropDialog(img);
});

async function duplicateImage(img) {
  setBusy(true);
  try {
    const data = await post("/api/source/duplicate", { image_id: img.image_id, job_id: img.job_id });
    if (data.error) throw new Error(data.error);
    images.push(data.image);
    renderList();
    selectImage(data.image);
  } catch (e) { alert(e.message); }
  finally { setBusy(false); }
}

function openCropDialog(img) {
  cropImageSource = img;
  cropImage.src = img.url;
  cropImage.onload = () => {
    cropSelection.classList.add("hidden");
    cropSelection.style.width = "0px";
    cropSelection.style.height = "0px";
  };
  cropSelection.classList.add("hidden");
  cropDialog.classList.remove("hidden");
}

function closeCropDialog() {
  cropDialog.classList.add("hidden");
  cropImageSource = null;
  cropStart = null;
}

function cropPoint(ev) {
  const rect = cropImage.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(rect.width, ev.clientX - rect.left)),
    y: Math.max(0, Math.min(rect.height, ev.clientY - rect.top))
  };
}

function cropImageOffset() {
  const imageRect = cropImage.getBoundingClientRect();
  const stageRect = cropImage.parentElement.getBoundingClientRect();
  return { x: imageRect.left - stageRect.left, y: imageRect.top - stageRect.top };
}

function updateCropSelection(start, end) {
  const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
  const offset = cropImageOffset();
  cropSelection.style.left = `${x + offset.x}px`;
  cropSelection.style.top = `${y + offset.y}px`;
  cropSelection.style.width = `${Math.abs(end.x - start.x)}px`;
  cropSelection.style.height = `${Math.abs(end.y - start.y)}px`;
}

cropStage.addEventListener("pointerdown", (ev) => {
  if (!cropImageSource || !cropImage.naturalWidth || ev.target !== cropImage) return;
  ev.preventDefault();
  cropStart = cropPoint(ev);
  cropSelection.classList.remove("hidden");
  cropStage.setPointerCapture(ev.pointerId);
  updateCropSelection(cropStart, cropStart);
});
cropStage.addEventListener("pointermove", (ev) => {
  if (cropStart) updateCropSelection(cropStart, cropPoint(ev));
});
cropStage.addEventListener("pointerup", (ev) => {
  if (cropStart) cropStage.releasePointerCapture(ev.pointerId);
  cropStart = null;
});
cropStage.addEventListener("pointercancel", () => { cropStart = null; });

$("#closeCrop").onclick = closeCropDialog;
$("#cancelCrop").onclick = closeCropDialog;
$("#applyCrop").onclick = async () => {
  if (!cropImageSource || !cropImage.naturalWidth || !cropSelection.offsetWidth || !cropSelection.offsetHeight) {
    alert("Drag across the image to choose a crop area.");
    return;
  }
  const scaleX = cropImage.naturalWidth / cropImage.clientWidth;
  const scaleY = cropImage.naturalHeight / cropImage.clientHeight;
  const offset = cropImageOffset();
  setBusy(true);
  try {
    const data = await post("/api/source/crop", {
      image_id: cropImageSource.image_id,
      job_id: cropImageSource.job_id,
      x: Math.round((parseFloat(cropSelection.style.left) - offset.x) * scaleX),
      y: Math.round((parseFloat(cropSelection.style.top) - offset.y) * scaleY),
      width: Math.round(cropSelection.offsetWidth * scaleX),
      height: Math.round(cropSelection.offsetHeight * scaleY)
    });
    if (data.error) throw new Error(data.error);
    images.push(data.image);
    closeCropDialog();
    renderList();
    selectImage(data.image);
  } catch (e) { alert(e.message); }
  finally { setBusy(false); }
}

function markProcessed(img, taskName) {
  const key = taskKey(img);
  const set = processedTasks.get(key) || new Set();
  set.add(taskName);
  processedTasks.set(key, set);
}

function hasProcessed(img, taskName) {
  return Boolean(processedTasks.get(taskKey(img))?.has(taskName));
}

function syncProcessedTasksFromHistory(items) {
  const next = new Map();
  for (const item of items) {
    const taskName = item.mode === "bg" ? "bg" : item.mode === "auto" ? "auto" : null;
    if (!taskName || !item.source_name) continue;
    const key = `${item.job_id || "single"}:${item.source_name}`;
    const set = next.get(key) || new Set();
    set.add(taskName);
    next.set(key, set);
  }
  processedTasks = next;
}

results.addEventListener("click", async (ev) => {
  const animateButton = ev.target.closest(".animate-result");
  if (animateButton) {
    const source = {
      file_name: animateButton.dataset.fileName,
      url: `/api/output/${animateButton.dataset.fileName}`,
      name: animateButton.dataset.label,
      mode: "auto",
      sourceType: "processed"
    };
    addAnimationSource(source);
    addAnimationLibrarySource(source);
    setMode("animator");
    return;
  }
  const saveButton = ev.target.closest(".save-auto-result");
  if (saveButton) {
    ev.preventDefault();
    ev.stopPropagation();
    saveAutoResult(saveButton.dataset.url, saveButton);
    return;
  }

  const deleteButton = ev.target.closest(".delete-result");
  if (deleteButton) {
    ev.preventDefault();
    ev.stopPropagation();
    await deleteProcessedResult(deleteButton.dataset.fileName, deleteButton.closest(".result"));
  }
});

$("#browseBtn").onclick = () => fileInput.click();
$("#addMore").onclick = () => fileInput.click();
fileInput.addEventListener("change", e => upload(e.target.files));

[dropZone, sidebar, sidebarDrop, imageList].forEach(zone => {
  ["dragenter", "dragover"].forEach(e =>
    zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach(e =>
    zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove("drag"); })
  );
  zone.addEventListener("drop", e => upload(e.dataTransfer.files));
});

$$(".tile[data-feature]").forEach(t => t.onclick = () => selectFeature(t.dataset.feature));
backHome.onclick = goHome;
$$(".mode").forEach(btn => btn.onclick = () => setMode(btn.dataset.mode));
$("#clearHistory").onclick = clearHistory;
animationInterval.addEventListener("input", () => {
  animationIntervalValue.textContent = `${Number(animationInterval.value).toFixed(1)} sec`;
  updateAnimationPreview();
});
buildAnimationButton.onclick = buildAnimation;
clearAnimationFrames.onclick = () => {
  animationSources = [];
  animationResult.innerHTML = "";
  animationPreview.removeAttribute("src");
  animationVideo.removeAttribute("src");
  renderAnimationLibrary();
  renderAnimationSources();
};

function selectFeature(feature) {
  mode = feature;
  home.classList.add("hidden");
  workspaceView.classList.remove("hidden");
  backHome.classList.remove("hidden");
  $$(".mode").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));

  if (images.length) {
    dropZone.classList.add("hidden");
    workspace.classList.remove("hidden");
    if (!current) selectImage(images[0]);
    else setMode(mode);
  } else {
    dropZone.classList.remove("hidden");
    workspace.classList.add("hidden");
  }
  if (feature === "animator") syncAnimationSourcesFromQueue();
}

function goHome() {
  workspaceView.classList.add("hidden");
  backHome.classList.add("hidden");
  home.classList.remove("hidden");
}

async function upload(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach(f => fd.append("files", f));

  setBusy(true);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    images.push(...data.images);
    animationLibrarySources.push(...data.images.map(img => ({
      image_id: img.image_id, job_id: img.job_id, name: img.name, url: img.url
    })));
    syncAnimationSourcesFromQueue();
    workspace.classList.remove("hidden");
    dropZone.classList.add("hidden");
    renderList();

    if (!current && images.length) selectImage(images[0]);
  } catch (e) {
    alert(e.message);
  } finally {
    setBusy(false);
  }
}

function renderList() {
  imageList.innerHTML = "";
  images.forEach(img => {
    const el = document.createElement("div");
    el.className = "thumb" + (current === img ? " active" : "");
    el.innerHTML = `
      <button class="thumb-remove" title="Remove from queue">×</button>
      <img src="${img.url}">
      <div>${escapeHtml(img.name)}</div>`;
    el.querySelector("img").onclick = () => selectImage(img);
    el.querySelector("div").onclick = () => selectImage(img);
    el.querySelector(".thumb-remove").onclick = (ev) => { ev.stopPropagation(); removeImage(img); };
    imageList.appendChild(el);
  });
}

function syncAnimationSourcesFromQueue() {
  const uploadedSources = images.map(img => ({ image_id: img.image_id, job_id: img.job_id, name: img.name, url: img.url, sourceType: "input" }));
  const allSources = [...animationLibrarySources, ...uploadedSources];
  animationLibrarySources = allSources.filter((source, index) =>
    isAnimatorSource(source) && allSources.findIndex(item => sourceKey(item) === sourceKey(source)) === index
  );
  renderAnimationLibrary();
  renderAnimationSources();
}

function isAnimatorSource(source) {
  const name = source.file_name || source.name || source.url || "";
  if (!animatorImageExtensions.test(name.split("?")[0])) return false;
  if (source.sourceType === "input" || source.image_id && !source.file_name) return true;
  return animatorHistoryModes.has(source.mode);
}

function sourceKey(source) {
  return source.file_name ? `file:${source.file_name}` : `image:${source.job_id || "single"}:${source.image_id}`;
}

function addAnimationLibrarySource(source) {
  if (!isAnimatorSource(source)) return;
  if (!animationLibrarySources.some(item => sourceKey(item) === sourceKey(source))) {
    animationLibrarySources.push(source);
  }
  renderAnimationLibrary();
}

function renderAnimationLibrary() {
  animationLibrary.innerHTML = animationLibrarySources.map(source => {
    const added = animationSources.some(item => sourceKey(item) === sourceKey(source));
    return `<button class="library-frame${added ? " added" : ""}" data-source-key="${escapeHtml(sourceKey(source))}" ${added ? "disabled" : ""}>
      <img src="${source.url}" alt="">
      <span>${escapeHtml(source.name || source.file_name || "Image")}</span>
      <b>${added ? "Added" : "+ Add"}</b>
    </button>`;
  }).join("") || `<div class="muted">Upload images or add Auto Extract results to see them here.</div>`;
  animationLibrary.querySelectorAll(".library-frame:not(.added)").forEach(button => {
    button.onclick = () => {
      const source = animationLibrarySources.find(item => sourceKey(item) === button.dataset.sourceKey);
      if (source) addAnimationSource(source);
    };
  });
}

function addAnimationSource(source) {
  if (!isAnimatorSource(source)) return;
  if (animationSources.some(item => sourceKey(item) === sourceKey(source))) return;
  animationSources.push(source);
  addAnimationLibrarySource(source);
  renderAnimationSources();
}

function renderAnimationSources() {
  animationFrames.innerHTML = animationSources.map((source, index) => `
    <div class="animation-frame">
      ${source.url ? `<img src="${source.url}" alt="">` : `<div class="frame-placeholder">${index + 1}</div>`}
      <span>${index + 1}. ${escapeHtml(source.name || source.file_name || "Frame")}</span>
      <button class="remove-animation-frame" data-index="${index}" title="Remove frame">×</button>
    </div>`).join("");
  animationFrames.querySelectorAll(".remove-animation-frame").forEach(button => {
    button.onclick = () => {
      animationSources.splice(Number(button.dataset.index), 1);
      renderAnimationLibrary();
      renderAnimationSources();
    };
  });
  updateAnimationPreview();
}

function updateAnimationPreview() {
  clearInterval(animationTimer);
  animationPreview.classList.add("hidden");
  animationVideo.classList.add("hidden");
  animationEmpty.classList.toggle("hidden", animationSources.length >= 2);
  if (animationSources.length < 2) return;
  const imageSources = animationSources.filter(source => source.url);
  if (imageSources.length < 2) {
    animationEmpty.textContent = "Create the animation to preview extracted frames.";
    return;
  }
  animationEmpty.classList.add("hidden");
  animationPreview.classList.remove("hidden");
  let index = 0;
  animationPreview.src = imageSources[0].url;
  animationTimer = setInterval(() => {
    index = (index + 1) % imageSources.length;
    animationPreview.src = imageSources[index].url;
  }, Number(animationInterval.value) * 1000);
}

imageList.addEventListener("contextmenu", (ev) => {
  const item = ev.target.closest(".thumb");
  if (!item || !imageList.contains(item)) return;
  ev.preventDefault();
  ev.stopPropagation();
  const index = [...imageList.children].indexOf(item);
  menuImage = images[index];
  queueMenu.style.left = `${Math.min(ev.clientX, window.innerWidth - 190)}px`;
  queueMenu.style.top = `${Math.min(ev.clientY, window.innerHeight - 100)}px`;
  queueMenu.classList.remove("hidden");
});

async function removeImage(img) {
  const url = img.job_id
    ? `/api/source/${img.job_id}/${img.image_id}`
    : `/api/source/${img.image_id}`;
  try { await fetch(url, { method: "DELETE" }); } catch (e) { /* still remove from queue */ }

  const idx = images.indexOf(img);
  if (idx >= 0) images.splice(idx, 1);
  processedTasks.delete(taskKey(img));

  if (current === img) {
    current = null;
    objects = [];
    activeIndex = -1;
    results.innerHTML = "";
    objectPanel.classList.add("hidden");
    if (images.length) {
      selectImage(images[0]);
    } else {
      mainImage.removeAttribute("src");
      overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
      workspace.classList.add("hidden");
      dropZone.classList.remove("hidden");
    }
  }
  renderList();
}

function resetClickUI() {
  objects = [];
  activeIndex = -1;
  objectPanel.classList.add("hidden");
  clickControls.classList.add("hidden");
  clickControls.innerHTML = "";
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function selectImage(img) {
  current = img;
  results.innerHTML = "";
  resetClickUI();
  mainImage.src = img.url;
  mainImage.onload = () => {
    resizeOverlay();
    renderList();
    canvasWrap.classList.toggle("clickable", mode === "click");
    syncRunButton();
    if (mode === "click") {
      drawObjects();
      hint.textContent = "Click an object to select it (Shift+click to remove an area). Use the panel below to manage selections.";
      objectPanel.classList.remove("hidden");
      renderObjectPanel();
      addPickerControls();
    }
  };
}

function syncRunButton() {
  const visible = Boolean(current) && (mode === "bg" || mode === "auto" || mode === "upscale");
  runCurrentTask.classList.toggle("hidden", !visible);
  animatorPanel.classList.toggle("hidden", mode !== "animator");
  canvasWrap.classList.toggle("hidden", mode === "animator");
  hint.classList.toggle("hidden", mode === "animator");
  upscaleControls.classList.toggle("hidden", mode !== "upscale");
  chromaControls.classList.toggle("hidden", mode !== "auto");

  if (!current) return;
  if (mode === "bg") runCurrentTask.textContent = "Run background removal";
  if (mode === "auto") runCurrentTask.textContent = "Run auto extract";
  if (mode === "upscale") runCurrentTask.textContent = "Run 2x upscale";
  if (mode === "animator") {
    syncAnimationSourcesFromQueue();
    renderAnimationSources();
  }
}

function setMode(newMode) {
  mode = newMode;
  pointMode = "foreground";
  $$(".mode").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
  canvasWrap.classList.toggle("clickable", mode === "click");

  if (mode !== "click") {
    resetClickUI();
  }

  if (mode === "bg") hint.textContent = "Choose a source image, then click Run background removal to process it.";
  if (mode === "click") hint.textContent = "Click to add a foreground point. Use background mode for exclusion points.";
  if (mode === "auto") hint.textContent = "Choose a source image, then click Run auto extract to detect objects.";
  if (mode === "upscale") hint.textContent = "Choose an uploaded image, then click Run 2x upscale to improve detail without processing it first.";
  if (mode === "animator") hint.textContent = "Arrange frames, set the interval, and create a GIF or MP4 animation.";

  syncRunButton();
  if (mode === "click") {
    drawObjects();
    hint.textContent = "Click an object to select it (Shift+click to remove an area). Use the panel below to manage selections.";
    objectPanel.classList.remove("hidden");
    renderObjectPanel();
    addPickerControls();
  }
}

runCurrentTask.onclick = () => {
  if (current) runCurrentMode();
};

function setPointMode(nextMode) {
  pointMode = nextMode === "background" ? "background" : "foreground";
  const btns = $$(".point-mode-btn");
  btns.forEach(btn => btn.classList.toggle("active", btn.dataset.pointMode === pointMode));
}

async function runCurrentMode() {
  if (!current) return;
  results.innerHTML = "";
  if (mode !== "click") {
    resetClickUI();
  }
  if (mode === "bg") {
    if (hasProcessed(current, "bg")) {
      results.innerHTML = "<div>Background already processed for this image.</div>";
      return;
    }
    await removeBG();
  }
  if (mode === "auto") {
    const settings = `${chromaKeyToggle.checked}:${chromaColor.value}:${borderThickness.value}:${chromaTolerance.value}`;
    if (hasProcessed(current, "auto") && autoRunSettings.get(taskKey(current)) === settings) {
      results.innerHTML = "<div>Auto extract already processed for this image.</div>";
      return;
    }
    autoRunSettings.set(taskKey(current), settings);
    await extractAll();
  }
  if (mode === "upscale") {
    await upscaleCurrent();
  }
  if (mode === "click") {
    drawObjects();
    hint.textContent = "Click an object to select it (Shift+click to remove an area). Use the panel below to manage selections.";
    objectPanel.classList.remove("hidden");
    renderObjectPanel();
    addPickerControls();
  }
}

async function upscaleCurrent() {
  setBusy(true);
  try {
    const upscaleMode = $("input[name='upscaleMode']:checked")?.value || "ai";
    const data = await post("/api/upscale", {
      image_id: current.image_id,
      job_id: current.job_id,
      mode: upscaleMode
    });
    if (data.error) throw new Error(data.error);
    if (data.sources?.length) {
      const sourceMap = new Map(animationSources.map((source, index) => [sourceKey(source), data.sources[index]]));
      animationSources = animationSources.map(source => sourceMap.get(sourceKey(source)) || source);
      animationLibrarySources = animationLibrarySources.map(source => sourceMap.get(sourceKey(source)) || source);
      renderAnimationLibrary();
      renderAnimationSources();
    }
    const url = data.url || "";
    if (!url) throw new Error("Upscale did not return an image URL.");
    results.innerHTML = resultCard(upscaleMode === "ai" ? "2x AI Upscaled" : "2x Smooth Upscaled", url);
    await loadHistory();
  } catch (e) {
    results.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
  } finally {
    setBusy(false);
  }
}

async function removeBG() {
  setBusy(true);
  try {
    const data = await post("/api/remove-background", {
      image_id: current.image_id, job_id: current.job_id
    });
    if (data.error) throw new Error(data.error);
    markProcessed(current, "bg");
    results.innerHTML = resultCard("Background removed", data.url);
    loadHistory();
  } catch (e) { results.innerHTML = `<div>${escapeHtml(e.message)}</div>`; }
  finally { setBusy(false); }
}

async function extractAll() {
  setBusy(true);
  autoExtractFiles = [];
  try {
    const data = await post("/api/extract-all", {
      image_id: current.image_id,
      job_id: current.job_id,
      chroma_key: chromaKeyToggle.checked,
      chroma_color: chromaColor.value,
      border_thickness: Number(borderThickness.value),
      chroma_tolerance: Number(chromaTolerance.value)
    });
    if (data.error) throw new Error(data.error);
    if (!data.objects.length) {
      markProcessed(current, "auto");
      results.innerHTML = "<div>No objects detected by the installed model.</div>";
      return;
    }
    markProcessed(current, "auto");
    autoExtractFiles = data.objects.map(object => object.url.split("/").pop()).filter(Boolean);
    results.innerHTML = data.objects.map(o => {
      const fileName = o.url.split("/").pop();
      return resultCard(`${o.label} · ${Math.round(o.confidence*100)}%`, o.url, true, o.label, fileName);
    }).join("");
  } catch (e) { results.innerHTML = `<div>${escapeHtml(e.message)}</div>`; }
  finally { setBusy(false); }
}

async function buildAnimation() {
  if (animationSources.length < 2) {
    animationResult.innerHTML = "<div>Add at least two frames first.</div>";
    return;
  }
  setBusy(true);
  buildAnimationButton.disabled = true;
  try {
    const data = await post("/api/animate", {
      sources: animationSources.map(source => ({
        image_id: source.image_id,
        job_id: source.job_id,
        file_name: source.file_name
      })),
      interval: Number(animationInterval.value),
      format: animationFormat.value,
      smooth: animationSmooth.checked
    });
    if (data.error) throw new Error(data.error);
    clearInterval(animationTimer);
    animationTimer = null;
    animationEmpty.classList.add("hidden");
    animationPreview.classList.add("hidden");
    animationVideo.classList.toggle("hidden", data.format !== "mp4");
    if (data.format === "mp4") {
      animationVideo.src = data.url;
      animationVideo.load();
      animationVideo.play().catch(() => {});
    } else {
      animationPreview.src = `${data.url}?t=${Date.now()}`;
      animationPreview.classList.remove("hidden");
    }
    animationResult.innerHTML = `<a class="primary download-animation" href="${data.url}" download>Download ${data.format.toUpperCase()}</a>`;
    await loadHistory();
  } catch (e) {
    animationResult.innerHTML = `<div>${escapeHtml(e.message)}</div>`;
  } finally {
    buildAnimationButton.disabled = false;
    setBusy(false);
  }
}

// ---- Object Picker (AI multi-point, multi-object) ----

function newObject() {
  objects.push({ points: [], maskUrl: null, bbox: null, tinted: null });
  activeIndex = objects.length - 1;
  pointMode = "foreground";
  renderObjectPanel();
  setPointMode(pointMode);
}

function clearActiveObject() {
  if (activeIndex < 0 || !objects[activeIndex]) return;
  objects[activeIndex].points = [];
  objects[activeIndex].maskUrl = null;
  objects[activeIndex].bbox = null;
  objects[activeIndex].tinted = null;
  drawObjects();
  renderObjectPanel();
}

async function resegmentObject(idx) {
  const obj = objects[idx];
  if (!obj || !obj.points.length) return;
  setBusy(true);
  try {
    const data = await post("/api/segment-click", {
      image_id: current.image_id,
      job_id: current.job_id,
      points: obj.points
    });
    if (data.error) throw new Error(data.error);
    obj.bbox = data.bbox;
    obj.maskUrl = data.mask_url;
    obj.tinted = null;
  } catch (e) {
    obj.points.pop();
    alert(e.message);
  } finally {
    setBusy(false);
    await drawObjects();
    renderObjectPanel();
  }
}

function undoPoint(idx) {
  const obj = objects[idx];
  if (!obj || !obj.points.length) return;
  obj.points.pop();
  if (!obj.points.length) {
    deleteObject(idx);
    return;
  }
  obj.tinted = null;
  resegmentObject(idx);
}

function deleteObject(idx) {
  objects.splice(idx, 1);
  if (activeIndex >= objects.length) activeIndex = objects.length - 1;
  drawObjects();
  renderObjectPanel();
}

overlay.addEventListener("click", async (ev) => {
  if (mode !== "click" || !current || segmentationInProgress) return;
  if (activeIndex === -1 || !objects[activeIndex]) newObject();

  const rect = mainImage.getBoundingClientRect();
  if (!rect.width || !rect.height || !mainImage.naturalWidth || !mainImage.naturalHeight) return;
  const x = Math.round((ev.clientX - rect.left) * mainImage.naturalWidth / rect.width);
  const y = Math.round((ev.clientY - rect.top) * mainImage.naturalHeight / rect.height);
  const safeX = Math.max(0, Math.min(x, mainImage.naturalWidth - 1));
  const safeY = Math.max(0, Math.min(y, mainImage.naturalHeight - 1));
  const label = pointMode === "background" ? 0 : 1;

  objects[activeIndex].points.push({ x: safeX, y: safeY, label });
  segmentationInProgress = true;
  try {
    await drawObjects();
    await resegmentObject(activeIndex);
  } finally {
    segmentationInProgress = false;
  }
});

function addPickerControls() {
  clickControls.innerHTML = "";
  clickControls.classList.remove("hidden");

  const newBtn = document.createElement("button");
  newBtn.className = "secondary";
  newBtn.textContent = "New object";
  newBtn.onclick = newObject;

  const clearBtn = document.createElement("button");
  clearBtn.className = "secondary";
  clearBtn.textContent = "Clear current object";
  clearBtn.onclick = clearActiveObject;

  const fgBtn = document.createElement("button");
  fgBtn.className = "secondary point-mode-btn active";
  fgBtn.dataset.pointMode = "foreground";
  fgBtn.textContent = "Foreground";
  fgBtn.onclick = () => setPointMode("foreground");

  const bgBtn = document.createElement("button");
  bgBtn.className = "secondary point-mode-btn";
  bgBtn.dataset.pointMode = "background";
  bgBtn.textContent = "Background";
  bgBtn.onclick = () => setPointMode("background");

  const exportBtn = document.createElement("button");
  exportBtn.className = "primary";
  exportBtn.textContent = "Export selected objects";
  exportBtn.onclick = exportSelected;

  clickControls.appendChild(newBtn);
  clickControls.appendChild(clearBtn);
  clickControls.appendChild(fgBtn);
  clickControls.appendChild(bgBtn);
  clickControls.appendChild(exportBtn);
}

async function exportSelected() {
  const usable = objects.filter(o => o.points.length);
  if (!usable.length) return;
  setBusy(true);
  try {
    const data = await post("/api/export-click", {
      image_id: current.image_id,
      job_id: current.job_id,
      objects: usable.map(o => ({ points: o.points }))
    });
    if (data.error) throw new Error(data.error);
    const cards = data.files.map((url, i) => resultCard(`Selected object ${i + 1}`, url)).join("");
    results.insertAdjacentHTML("beforeend", cards);
    loadHistory();
  } catch (e) { alert(e.message); }
  finally { setBusy(false); }
}

function renderObjectPanel() {
  objectList.innerHTML = objects.map((obj, idx) => `
    <div class="object-card${idx === activeIndex ? " active" : ""}" data-idx="${idx}">
      ${obj.maskUrl ? `<img src="${obj.maskUrl}">` : `<div class="object-placeholder">…</div>`}
      <div class="meta">Object ${idx + 1} · ${obj.points.length} pt${obj.points.length === 1 ? "" : "s"}</div>
      <div class="object-actions">
        <button class="undo-point">Undo point</button>
        <button class="danger delete-object">Delete</button>
      </div>
    </div>
  `).join("") || "<p class=\"muted\">No objects yet — click on the image to start one.</p>";

  $$(".object-card").forEach(card => {
    const idx = Number(card.dataset.idx);
    card.onclick = () => { activeIndex = idx; renderObjectPanel(); };
    card.querySelector(".undo-point").onclick = (ev) => { ev.stopPropagation(); undoPoint(idx); };
    card.querySelector(".delete-object").onclick = (ev) => { ev.stopPropagation(); deleteObject(idx); };
  });
}

async function drawObjects() {
  resizeOverlay();
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const rect = mainImage.getBoundingClientRect();

  for (const obj of objects) {
    if (!obj.maskUrl) continue;
    if (!obj.tinted) {
      try { obj.tinted = await tintMask(obj.maskUrl, "#5eead4"); } catch (e) { continue; }
    }
    const bbox = obj.bbox || [0, 0, mainImage.naturalWidth, mainImage.naturalHeight];
    const x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
    const drawX = (x1 / mainImage.naturalWidth) * rect.width;
    const drawY = (y1 / mainImage.naturalHeight) * rect.height;
    const drawW = ((x2 - x1 + 1) / mainImage.naturalWidth) * rect.width;
    const drawH = ((y2 - y1 + 1) / mainImage.naturalHeight) * rect.height;

    ctx.globalAlpha = 0.45;
    ctx.drawImage(obj.tinted, drawX, drawY, drawW, drawH);
    ctx.globalAlpha = 1;
  }

  objects.forEach(obj => {
    obj.points.forEach(p => {
      const x = (p.x / mainImage.naturalWidth) * rect.width;
      const y = (p.y / mainImage.naturalHeight) * rect.height;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? "#f87171" : "#5eead4";
      ctx.fill();
      ctx.strokeStyle = "#0b1120";
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  });
}

function tintMask(url, colorHex) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const cx = c.getContext("2d");
      cx.drawImage(img, 0, 0);
      const data = cx.getImageData(0, 0, c.width, c.height);
      const v = parseInt(colorHex.slice(1), 16);
      const r = (v >> 16) & 255, g = (v >> 8) & 255, b = v & 255;
      for (let i = 0; i < data.data.length; i += 4) {
        const lum = data.data[i];
        data.data[i] = r;
        data.data[i + 1] = g;
        data.data[i + 2] = b;
        data.data[i + 3] = lum;
      }
      cx.putImageData(data, 0, 0);
      resolve(c);
    };
    img.onerror = reject;
    img.src = url;
  });
}

// ---- History (last 10 processed results) ----

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const items = data.history || [];
    const processedSources = items
      .filter(item => animatorHistoryModes.has(item.mode) && animatorImageExtensions.test(item.filename || ""))
      .map(item => ({
        file_name: item.filename,
        name: item.source_name || item.filename,
        url: item.url,
        mode: item.mode,
        sourceType: "processed"
      }));
    const currentFrames = animationSources;
    animationLibrarySources = [...currentFrames, ...processedSources, ...images.map(img => ({
      image_id: img.image_id, job_id: img.job_id, name: img.name, url: img.url, sourceType: "input"
    }))].filter((source, index, all) =>
      isAnimatorSource(source) && all.findIndex(item => sourceKey(item) === sourceKey(source)) === index
    );
    animationSources = currentFrames.filter(isAnimatorSource);
    renderAnimationLibrary();
    renderAnimationSources();
    historyPanel.classList.toggle("hidden", items.length === 0);
    syncProcessedTasksFromHistory(items);
    historyList.innerHTML = items.map(h => `
      <div class="history-card" data-id="${h.id}">
        <img src="${h.url}">
        <div class="meta">${escapeHtml(h.mode)} · ${escapeHtml(shortId(h.source_name || ""))}</div>
        <div class="history-actions">
          <a href="${h.url}" download><button>Download</button></a>
          <button class="danger history-delete">Delete</button>
        </div>
      </div>
    `).join("");
    historyList.querySelectorAll(".history-delete").forEach(btn => {
      btn.onclick = async (ev) => {
        const id = ev.target.closest(".history-card").dataset.id;
        await fetch(`/api/history/${id}`, { method: "DELETE" });
        loadHistory();
      };
    });
  } catch (e) { /* history is best-effort */ }
}

async function clearHistory() {
  if (!confirm("Delete all history items? This cannot be undone.")) return;
  await fetch("/api/history", { method: "DELETE" });
  loadHistory();
}

function resizeOverlay() {
  const imageRect = mainImage.getBoundingClientRect();
  const wrapRect = canvasWrap.getBoundingClientRect();
  overlay.width = imageRect.width;
  overlay.height = imageRect.height;
  overlay.style.width = imageRect.width + "px";
  overlay.style.height = imageRect.height + "px";
  overlay.style.left = imageRect.left - wrapRect.left + "px";
  overlay.style.top = imageRect.top - wrapRect.top + "px";
}

window.addEventListener("resize", resizeOverlay);

function setBusy(v) { loading.classList.toggle("hidden", !v); }

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return await res.json();
}

function resultCard(title, url, showAnimate = false, label = "Frame", outputFileName = "") {
  const fileName = url.split("/").pop() || outputFileName;
  return `<div class="result">
    <img src="${url}">
    <div class="meta">${escapeHtml(title)}</div>
    <div class="result-actions">
      <a href="${url}" download><button>Download</button></a>
      <button class="save-auto-result" data-url="${escapeHtml(url)}">Save</button>
      ${showAnimate ? `<button class="animate-result" data-file-name="${escapeHtml(fileName)}" data-label="${escapeHtml(label)}">+ Add frame</button>` : ""}
      <button class="danger delete-result" data-file-name="${escapeHtml(fileName)}">Delete</button>
    </div>
  </div>`;
}

async function saveAutoResult(url, button) {
  const name = url.split("/").pop();
  if (!name) return;
  if (button?.disabled) return;
  if (button) {
    button.disabled = true;
    button.textContent = "Saving...";
  }
  try {
    const res = await post("/api/save-selected", {
      image_id: current?.image_id,
      files: [name],
      preserve_files: autoExtractFiles
    });
    if (res.error) throw new Error(res.error);
    if (button) button.textContent = "Saved";
    await loadHistory();
  } catch (e) {
    if (button) {
      button.disabled = false;
      button.textContent = "Save";
    }
    alert(e.message);
  }
}

async function deleteProcessedResult(fileName, card) {
  if (!fileName || !confirm("Delete this processed image?")) return;
  try {
    const response = await fetch(`/api/output/${encodeURIComponent(fileName)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Delete failed");
    card?.remove();
    await loadHistory();
  } catch (e) {
    alert(e.message);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])
  );
}

loadHistory();
