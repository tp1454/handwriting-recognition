const form = document.querySelector("#score-form");
const fileInput = document.querySelector("#sheet-file");
const statusNode = document.querySelector("#status");
const submitButton = document.querySelector("#submit-btn");
const previewImage = document.querySelector("#preview-image");
const previewCanvas = document.querySelector("#preview-canvas");
const boxOverlay = document.querySelector("#box-overlay");
const previewEmpty = document.querySelector("#preview-empty");
const showOverlayCheckbox = document.querySelector("#show-overlay");
const summaryNode = document.querySelector("#summary");
const tableWrap = document.querySelector("#table-wrap");
const rowsBody = document.querySelector("#rows-body");
const resultsEmpty = document.querySelector("#results-empty");

const apiBaseUrl = (window.HW_APP_CONFIG?.apiBaseUrl || "http://localhost:8000").replace(/\/$/, "");
let lastExtractedBoxes = [];

function clearOverlay() {
  if (boxOverlay) {
    boxOverlay.innerHTML = "";
    boxOverlay.classList.add("hidden");
  }
}

function drawOverlay(extractedBoxes) {
  if (!boxOverlay || !previewImage || !previewCanvas) {
    return;
  }

  clearOverlay();

  if (!showOverlayCheckbox?.checked || !Array.isArray(extractedBoxes) || extractedBoxes.length === 0) {
    return;
  }

  const naturalWidth = previewImage.naturalWidth || 1;
  const naturalHeight = previewImage.naturalHeight || 1;
  const renderedWidth = previewImage.clientWidth || 1;
  const renderedHeight = previewImage.clientHeight || 1;
  const ratioX = renderedWidth / naturalWidth;
  const ratioY = renderedHeight / naturalHeight;

  boxOverlay.classList.remove("hidden");
  extractedBoxes.forEach((item) => {
    const box = Array.isArray(item?.box) ? item.box : [];
    if (box.length !== 4) {
      return;
    }

    const [xMin, xMax, yMin, yMax] = box.map((value) => Number(value));
    const width = Math.max(1, (xMax - xMin) * ratioX);
    const height = Math.max(1, (yMax - yMin) * ratioY);

    const node = document.createElement("div");
    node.className = "overlay-box";
    node.style.left = `${xMin * ratioX}px`;
    node.style.top = `${yMin * ratioY}px`;
    node.style.width = `${width}px`;
    node.style.height = `${height}px`;

    const label = document.createElement("span");
    label.className = "overlay-label";
    label.textContent = String(item?.label || "UNKNOWN");
    node.appendChild(label);
    boxOverlay.appendChild(node);
  });
}

function setStatus(message, mode = "info") {
  statusNode.textContent = message;
  statusNode.classList.remove("error", "warning");
  if (mode === "error") {
    statusNode.classList.add("error");
  }
  if (mode === "warning") {
    statusNode.classList.add("warning");
  }
}

function resetResults() {
  summaryNode.classList.add("hidden");
  tableWrap.classList.add("hidden");
  resultsEmpty.classList.remove("hidden");
  rowsBody.innerHTML = "";
  lastExtractedBoxes = [];
  clearOverlay();
}

function renderResults(payload) {
  const overall = Number(payload.overall_score || 0).toFixed(2);
  summaryNode.innerHTML = `<strong>Overall score:</strong> ${overall}/100`;
  summaryNode.classList.remove("hidden");

  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  rowsBody.innerHTML = rows
    .map((row) => {
      const segmentScores = Array.isArray(row.segment_scores) ? row.segment_scores : [];
      const scores = segmentScores.length > 0
        ? segmentScores.map((score) => Number(score).toFixed(1)).join(", ")
        : "-";
      return `
      <tr>
        <td>${row.row_index + 1}</td>
        <td>${String(row.ocr_label || "UNKNOWN")}</td>
        <td>${row.split_count}</td>
        <td><span class="score-pill">${Number(row.row_score).toFixed(2)}</span></td>
        <td>${scores}</td>
      </tr>
    `;
    })
    .join("");

  lastExtractedBoxes = Array.isArray(payload.extracted_boxes) ? payload.extracted_boxes : [];
  drawOverlay(lastExtractedBoxes);

  tableWrap.classList.remove("hidden");
  resultsEmpty.classList.add("hidden");
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    previewImage.style.display = "none";
    previewImage.removeAttribute("src");
    previewEmpty.classList.remove("hidden");
    clearOverlay();
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = String(event.target?.result || "");
    previewImage.style.display = "block";
    previewEmpty.classList.add("hidden");
    drawOverlay(lastExtractedBoxes);
  };
  reader.readAsDataURL(file);
});

previewImage.addEventListener("load", () => {
  drawOverlay(lastExtractedBoxes);
});

showOverlayCheckbox?.addEventListener("change", () => {
  drawOverlay(lastExtractedBoxes);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetResults();

  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Select an image file before submitting.", "warning");
    return;
  }

  submitButton.disabled = true;
  setStatus("Scoring sheet...");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${apiBaseUrl}/sheet/score`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload?.detail || "Request failed";
      throw new Error(String(detail));
    }

    renderResults(payload);
    setStatus("Scoring complete.");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Unexpected error", "error");
  } finally {
    submitButton.disabled = false;
  }
});
