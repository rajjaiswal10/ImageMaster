const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const home = $("#home");
const workspaceView = $("#workspaceView");
const backHome = $("#backHome");
const dropZone = $("#dropZone");
const sidebarDrop = $("#sidebarDrop");
const fileInput = $("#fileInput");
const workspace = $("#workspace");
const imageList = $("#imageList");
const mainImage = $("#mainImage");
const overlay = $("#overlay");
const canvasWrap = $("#canvasWrap");
const loading = $("#loading");
const results = $("#results");
const hint = $("#hint");
const objectPanel = $("#objectPanel");
const objectList = $("#objectList");
const historyPanel = $("#historyPanel");
const historyList = $("#historyList");

let images = [];
let current = null;
let mode = "bg";
let objects = [];        // click-mode selections: [{ points:[{x,y,label}], maskUrl, bbox, tinted }]
let activeIndex = -1;

$("#browseBtn").onclick = () => fileInput.click();
$("#addMore").onclick = () => fileInput.click();
fileInput.addEventListener("change", e => upload(e.target.files));

[dropZone, sidebarDrop].forEach(zone => {
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
    else runCurrentMode();
  } else {
    dropZone.classList.remove("hidden");
    workspace.classList.add("hidden");
  }
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

async function removeImage(img) {
  const url = img.job_id
    ? `/api/source/${img.job_id}/${img.image_id}`
    : `/api/source/${img.image_id}`;
  try { await fetch(url, { method: "DELETE" }); } catch (e) { /* still remove from queue */ }

  const idx = images.indexOf(img);
  if (idx >= 0) images.splice(idx, 1);

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

function selectImage(img) {
  current = img;
  objects = [];
  activeIndex = -1;
  results.innerHTML = "";
  objectPanel.classList.add("hidden");
  mainImage.src = img.url;
  mainImage.onload = () => {
    resizeOverlay();
    renderList();
    runCurrentMode();
  };
}

function setMode(newMode) {
  mode = newMode;
  $$(".mode").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
  canvasWrap.classList.toggle("clickable", mode === "click");

  if (mode === "bg") hint.textContent = "Automatic background removal with alpha matting.";
  if (mode === "click") hint.textContent = "Click an object to select it (Shift+click to remove an area from the selection).";
  if (mode === "auto") hint.textContent = "AI segmentation detects multiple supported objects and exports separate transparent PNGs.";

  if (current) runCurrentMode();
}

async function runCurrentMode() {
  if (!current) return;
  results.innerHTML = "";
  objects = [];
  activeIndex = -1;
  objectPanel.classList.add("hidden");
  if (mode === "bg") await removeBG();
  if (mode === "auto") await extractAll();
  if (mode === "click") {
    drawObjects();
    hint.textContent = "Click an object to select it (Shift+click to remove an area). Use the panel below to manage selections.";
    objectPanel.classList.remove("hidden");
    renderObjectPanel();
    addPickerControls();
  }
}

async function removeBG() {
  setBusy(true);
  try {
    const data = await post("/api/remove-background", {
      image_id: current.image_id, job_id: current.job_id
    });
    if (data.error) throw new Error(data.error);
    results.innerHTML = resultCard("Background removed", data.url);
    loadHistory();
  } catch (e) { results.innerHTML = `<div>${escapeHtml(e.message)}</div>`; }
  finally { setBusy(false); }
}

async function extractAll() {
  setBusy(true);
  try {
    const data = await post("/api/extract-all", {
      image_id: current.image_id, job_id: current.job_id
    });
    if (data.error) throw new Error(data.error);
    if (!data.objects.length) {
      results.innerHTML = "<div>No objects detected by the installed model.</div>";
      return;
    }
    results.innerHTML = data.objects.map(o =>
      resultCard(`${o.label} · ${Math.round(o.confidence*100)}%`, o.url)
    ).join("");
    loadHistory();
  } catch (e) { results.innerHTML = `<div>${escapeHtml(e.message)}</div>`; }
  finally { setBusy(false); }
}

// ---- Object Picker (AI multi-point, multi-object) ----

function newObject() {
  objects.push({ points: [], maskUrl: null, bbox: null, tinted: null });
  activeIndex = objects.length - 1;
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
  if (mode !== "click" || !current) return;
  if (activeIndex === -1 || !objects[activeIndex]) newObject();

  const rect = mainImage.getBoundingClientRect();
  const x = Math.round((ev.clientX - rect.left) * mainImage.naturalWidth / rect.width);
  const y = Math.round((ev.clientY - rect.top) * mainImage.naturalHeight / rect.height);
  const label = ev.shiftKey ? 0 : 1;

  objects[activeIndex].points.push({ x, y, label });
  await drawObjects();
  await resegmentObject(activeIndex);
});

function addPickerControls() {
  const newBtn = document.createElement("button");
  newBtn.className = "secondary";
  newBtn.textContent = "New object";
  newBtn.onclick = newObject;

  const exportBtn = document.createElement("button");
  exportBtn.className = "primary";
  exportBtn.textContent = "Export selected objects";
  exportBtn.onclick = exportSelected;

  results.appendChild(newBtn);
  results.appendChild(exportBtn);
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
    ctx.globalAlpha = 0.45;
    ctx.drawImage(obj.tinted, 0, 0, rect.width, rect.height);
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
    historyPanel.classList.toggle("hidden", items.length === 0);
    historyList.innerHTML = items.map(h => `
      <div class="history-card" data-id="${h.id}">
        <img src="${h.url}">
        <div class="meta">${escapeHtml(h.mode)} · ${escapeHtml(h.source_name || "")}</div>
        <div class="history-actions">
          <button class="post-process" disabled title="Coming soon">Post-process ▾</button>
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

function resultCard(title, url) {
  return `<div class="result">
    <img src="${url}">
    <div class="meta">${escapeHtml(title)}</div>
    <div class="result-actions">
      <a href="${url}" download><button>Download</button></a>
      <button class="post-process" disabled title="Coming soon">Post-process ▾</button>
    </div>
  </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])
  );
}

loadHistory();
