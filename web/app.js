const form = document.querySelector("#score-form");
const fileInput = document.querySelector("#sheet-file");
const statusNode = document.querySelector("#status");
const submitButton = document.querySelector("#submit-btn");
const previewImage = document.querySelector("#preview-image");
const boxOverlay = document.querySelector("#box-overlay");
const previewEmpty = document.querySelector("#preview-empty");
const summaryNode = document.querySelector("#summary");
const tableWrap = document.querySelector("#table-wrap");
const rowsBody = document.querySelector("#rows-body");
const resultsEmpty = document.querySelector("#results-empty");
const createOutputNode = document.querySelector("#create-output");
const createMetaNode = document.querySelector("#create-meta");
const previewLinkNode = document.querySelector("#preview-link");
const pdfLinkNode = document.querySelector("#pdf-link");
const pngLinkNode = document.querySelector("#png-link");

const scoreFields = document.querySelector("#score-fields");
const createFields = document.querySelector("#create-fields");
const modeInputs = Array.from(document.querySelectorAll("input[name='mode']"));
const overlayModeInputs = Array.from(document.querySelectorAll("input[name='overlay-mode']"));
const fontSourceInputs = Array.from(document.querySelectorAll("input[name='font-source']"));
const serverFontField = document.querySelector("#server-font-field");
const fontUploadField = document.querySelector("#font-upload-field");
const serverFontSelect = document.querySelector("#server-font");
const serverFontHelpNode = document.querySelector("#server-font-help");
const fontFileInput = document.querySelector("#font-file");
const createLanguageSelect = document.querySelector("#create-language");
const customTextInput = document.querySelector("#custom-text");
const outputBasenameInput = document.querySelector("#output-basename");

const numericFieldMap = {
  font_size: document.querySelector("#font-size"),
  line_spacing: document.querySelector("#line-spacing"),
  word_spacing: document.querySelector("#word-spacing"),
  margin_left: document.querySelector("#margin-left"),
  margin_right: document.querySelector("#margin-right"),
  margin_top: document.querySelector("#margin-top"),
  margin_bottom: document.querySelector("#margin-bottom"),
  divide_horizontal: document.querySelector("#divide-horizontal"),
  divide_vertical: document.querySelector("#divide-vertical"),
};
const showVerticalLineInput = document.querySelector("#show-vertical-line");

const languageLabels = {
  auto: "Auto detect from font",
  en: "English",
  vi: "Vietnamese",
};

const apiBaseUrl = (window.HW_APP_CONFIG?.apiBaseUrl || "http://localhost:8000").replace(/\/$/, "");
let lastExtractedBoxes = [];
let expandedScoreRowKey = null;

function selectedMode() {
  const selected = modeInputs.find((input) => input.checked);
  return selected?.value || "score";
}

function selectedFontSource() {
  const selected = fontSourceInputs.find((input) => input.checked);
  return selected?.value || "server";
}

function selectedOverlayMode() {
  const selected = overlayModeInputs.find((input) => input.checked);
  return selected?.value || "off";
}

function setOverlayMode(mode) {
  const selected = overlayModeInputs.find((input) => input.value === mode);
  if (selected) {
    selected.checked = true;
  }
}

function toArray(value, fallback = []) {
  return Array.isArray(value) ? value : fallback;
}

function toTrimmedString(value, fallback = "") {
  const normalized = String(value || "").trim();
  return normalized || fallback;
}

function toNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function normalizeApiUrl(value) {
  const raw = toTrimmedString(value);
  if (!raw) {
    return "";
  }
  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }
  return `${apiBaseUrl}${raw.startsWith("/") ? "" : "/"}${raw}`;
}

async function parseResponsePayload(response) {
  try {
    return await response.json();
  } catch (_error) {
    return {};
  }
}

function clearOverlay() {
  if (boxOverlay) {
    boxOverlay.innerHTML = "";
    boxOverlay.classList.add("hidden");
  }
}

function drawOverlay(extractedBoxes) {
  if (!boxOverlay || !previewImage) {
    return;
  }

  clearOverlay();

  if (selectedMode() !== "score") {
    return;
  }

  if (selectedOverlayMode() !== "boxes" || !Array.isArray(extractedBoxes) || extractedBoxes.length === 0) {
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

function clearPreview() {
  previewImage.style.display = "none";
  previewImage.removeAttribute("src");
  previewEmpty.classList.remove("hidden");
  clearOverlay();
}

function setPreviewFromUrl(url) {
  const normalized = normalizeApiUrl(url);
  if (!normalized) {
    clearPreview();
    return;
  }
  previewImage.src = normalized;
  previewImage.style.display = "block";
  previewEmpty.classList.add("hidden");
}

function resetResults() {
  summaryNode.classList.add("hidden");
  summaryNode.innerHTML = "";
  tableWrap.classList.add("hidden");
  rowsBody.innerHTML = "";
  expandedScoreRowKey = null;
  createOutputNode.classList.add("hidden");
  createMetaNode.textContent = "";
  previewLinkNode.removeAttribute("href");
  pdfLinkNode.removeAttribute("href");
  pngLinkNode.removeAttribute("href");
  resultsEmpty.classList.remove("hidden");
  lastExtractedBoxes = [];
  clearOverlay();
}

function buildImageDataUri(base64Payload, mimeType = "image/png") {
  const imageData = toTrimmedString(base64Payload);
  if (!imageData) {
    return "";
  }
  const normalizedMimeType = toTrimmedString(mimeType, "image/png");
  return `data:${normalizedMimeType};base64,${imageData}`;
}

function formatBox(box) {
  if (!Array.isArray(box) || box.length !== 4) {
    return "[unknown]";
  }
  return `[${box.map((value) => Math.round(toNumber(value, 0))).join(", ")}]`;
}

function createTextCell(value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  return cell;
}

function collapseExpandedScoreRow() {
  if (!expandedScoreRowKey) {
    return;
  }

  const activeSummary = rowsBody.querySelector(
    `tr.score-row[data-row-key="${expandedScoreRowKey}"]`,
  );
  const activeDetail = rowsBody.querySelector(
    `tr.score-details-row[data-row-key="${expandedScoreRowKey}"]`,
  );

  if (activeSummary) {
    activeSummary.classList.remove("expanded");
    activeSummary.setAttribute("aria-expanded", "false");
  }
  if (activeDetail) {
    activeDetail.classList.add("hidden");
  }

  expandedScoreRowKey = null;
}

function toggleScoreRowDetails(summaryRow, detailRow) {
  const rowKey = String(summaryRow.dataset.rowKey || "");
  if (!rowKey) {
    return;
  }

  const sameRowAlreadyExpanded = (
    expandedScoreRowKey === rowKey
    && !detailRow.classList.contains("hidden")
  );

  collapseExpandedScoreRow();

  if (sameRowAlreadyExpanded) {
    return;
  }

  summaryRow.classList.add("expanded");
  summaryRow.setAttribute("aria-expanded", "true");
  detailRow.classList.remove("hidden");
  expandedScoreRowKey = rowKey;
}

function createCharacterScoreCard(character, index, mimeType) {
  const card = document.createElement("article");
  card.className = "char-score-card";

  const header = document.createElement("div");
  header.className = "char-score-header";

  const splitIndex = toNumber(character?.split_index, index);
  const splitLabel = document.createElement("span");
  splitLabel.className = "char-score-split";
  splitLabel.textContent = character?.is_reference
    ? `Split ${splitIndex + 1} (ref)`
    : `Split ${splitIndex + 1}`;

  const scorePill = document.createElement("span");
  scorePill.className = "score-pill char-score-pill";
  scorePill.textContent = toNumber(
    character?.similarity_to_reference,
    0,
  ).toFixed(1);

  header.appendChild(splitLabel);
  header.appendChild(scorePill);
  card.appendChild(header);

  const imageUri = buildImageDataUri(character?.image, mimeType);
  if (imageUri) {
    const image = document.createElement("img");
    image.className = "char-score-image";
    image.src = imageUri;
    image.alt = `Split ${splitIndex + 1} extracted image`;
    card.appendChild(image);
  } else {
    const emptyImage = document.createElement("p");
    emptyImage.className = "detail-empty";
    emptyImage.textContent = "No extracted split image";
    card.appendChild(emptyImage);
  }

  const labelNode = document.createElement("p");
  labelNode.className = "char-score-label";
  labelNode.textContent = `OCR: ${toTrimmedString(character?.label, "UNKNOWN")}`;
  card.appendChild(labelNode);

  const boxNode = document.createElement("p");
  boxNode.className = "char-score-box";
  boxNode.textContent = `Box: ${formatBox(character?.box)}`;
  card.appendChild(boxNode);

  return card;
}

function createRowDetailsPanel(row) {
  const panel = document.createElement("div");
  panel.className = "score-detail-panel";

  const title = document.createElement("p");
  title.className = "score-detail-title";
  title.textContent = `Row ${toNumber(row.row_index, 0) + 1} details`;
  panel.appendChild(title);

  const details = row?.details && typeof row.details === "object"
    ? row.details
    : null;

  if (!details) {
    const fallback = document.createElement("p");
    fallback.className = "detail-empty";
    fallback.textContent = "Detailed row data is not available for this result.";
    panel.appendChild(fallback);
    return panel;
  }

  const rowImageSection = document.createElement("section");
  rowImageSection.className = "row-image-section";
  const rowImageLabel = document.createElement("p");
  rowImageLabel.className = "detail-section-label";
  rowImageLabel.textContent = "Extracted row image";
  rowImageSection.appendChild(rowImageLabel);

  const rowImageUri = buildImageDataUri(
    details?.row_image,
    details?.row_image_mime_type || "image/png",
  );
  if (rowImageUri) {
    const rowImage = document.createElement("img");
    rowImage.className = "row-detail-image";
    rowImage.src = rowImageUri;
    rowImage.alt = `Extracted row ${toNumber(row.row_index, 0) + 1}`;
    rowImageSection.appendChild(rowImage);
  } else {
    const rowImageFallback = document.createElement("p");
    rowImageFallback.className = "detail-empty";
    rowImageFallback.textContent = "No extracted row image";
    rowImageSection.appendChild(rowImageFallback);
  }
  panel.appendChild(rowImageSection);

  const charsSection = document.createElement("section");
  charsSection.className = "char-scores-section";
  const charsLabel = document.createElement("p");
  charsLabel.className = "detail-section-label";
  charsLabel.textContent = "Character scoring";
  charsSection.appendChild(charsLabel);

  const characters = toArray(details?.characters);
  if (characters.length === 0) {
    const emptyChars = document.createElement("p");
    emptyChars.className = "detail-empty";
    emptyChars.textContent = "No character score details returned.";
    charsSection.appendChild(emptyChars);
  } else {
    const grid = document.createElement("div");
    grid.className = "char-score-grid";
    characters.forEach((character, index) => {
      grid.appendChild(
        createCharacterScoreCard(
          character,
          index,
          details?.row_image_mime_type || "image/png",
        ),
      );
    });
    charsSection.appendChild(grid);
  }
  panel.appendChild(charsSection);

  return panel;
}

function renderScoreResults(payload) {
  const overall = Number(payload.overall_score || 0).toFixed(2);
  summaryNode.innerHTML = `<strong>Overall score:</strong> ${overall}/100`;
  summaryNode.classList.remove("hidden");

  const rows = toArray(payload.rows);
  rowsBody.innerHTML = "";
  expandedScoreRowKey = null;

  rows.forEach((row, index) => {
    const rowIndex = toNumber(row.row_index, index);
    const segmentScores = toArray(row.segment_scores);
    const splitCount = toNumber(row.split_count, 0);
    const comparedCount = Math.max(splitCount - 1, 0);
    const scores = segmentScores.length > 0
      ? segmentScores.map((score) => toNumber(score, 0).toFixed(1)).join(", ")
      : "-";

    const rowKey = String(rowIndex);
    const summaryRow = document.createElement("tr");
    summaryRow.className = "score-row";
    summaryRow.dataset.rowKey = rowKey;
    summaryRow.tabIndex = 0;
    summaryRow.setAttribute("role", "button");
    summaryRow.setAttribute("aria-expanded", "false");

    summaryRow.appendChild(createTextCell(String(rowIndex + 1)));
    summaryRow.appendChild(
      createTextCell(toTrimmedString(row.ocr_label, "UNKNOWN")),
    );
    summaryRow.appendChild(createTextCell(String(comparedCount)));

    const scoreCell = document.createElement("td");
    const scorePill = document.createElement("span");
    scorePill.className = "score-pill";
    scorePill.textContent = toNumber(row.row_score, 0).toFixed(2);
    scoreCell.appendChild(scorePill);
    summaryRow.appendChild(scoreCell);

    summaryRow.appendChild(createTextCell(scores));

    const detailRow = document.createElement("tr");
    detailRow.className = "score-details-row hidden";
    detailRow.dataset.rowKey = rowKey;
    const detailCell = document.createElement("td");
    detailCell.colSpan = 5;
    detailCell.appendChild(createRowDetailsPanel(row));
    detailRow.appendChild(detailCell);

    const toggleDetails = () => {
      toggleScoreRowDetails(summaryRow, detailRow);
    };

    summaryRow.addEventListener("click", toggleDetails);
    summaryRow.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggleDetails();
      }
    });

    rowsBody.appendChild(summaryRow);
    rowsBody.appendChild(detailRow);
  });

  lastExtractedBoxes = toArray(payload.extracted_boxes);
  drawOverlay(lastExtractedBoxes);

  tableWrap.classList.remove("hidden");
  createOutputNode.classList.add("hidden");
  resultsEmpty.classList.add("hidden");
}

function renderCreateResult(payload) {
  const language = String(payload?.language || "en").toUpperCase();
  const fontName = String(payload?.font_name || "Unknown font");
  const fontSource = String(payload?.font_source || "server");

  const previewUrl = normalizeApiUrl(payload?.preview_image_url);
  const pdfUrl = normalizeApiUrl(payload?.pdf_url);

  summaryNode.innerHTML = `<strong>Generated sheet:</strong> ${language} with ${fontName}`;
  summaryNode.classList.remove("hidden");

  createMetaNode.textContent = `Font source: ${fontSource} | Language: ${language}`;
  previewLinkNode.href = previewUrl;
  pdfLinkNode.href = pdfUrl;
  pngLinkNode.href = previewUrl;
  createOutputNode.classList.remove("hidden");

  tableWrap.classList.add("hidden");
  resultsEmpty.classList.add("hidden");
  clearOverlay();
  lastExtractedBoxes = [];
  setPreviewFromUrl(previewUrl);
}

function setPanelControlsEnabled(panelNode, enabled) {
  if (!panelNode) {
    return;
  }

  const controls = panelNode.querySelectorAll("input, select, textarea, button");
  controls.forEach((control) => {
    control.disabled = !enabled;
  });
}

function updateModeUI() {
  const scoreMode = selectedMode() === "score";
  scoreFields.classList.toggle("hidden", !scoreMode);
  createFields.classList.toggle("hidden", scoreMode);
  setPanelControlsEnabled(scoreFields, scoreMode);
  setPanelControlsEnabled(createFields, !scoreMode);
  submitButton.textContent = scoreMode ? "Score Sheet" : "Create Sheet";
  fileInput.required = scoreMode;
  updateFontSourceUI();

  if (!scoreMode) {
    setOverlayMode("off");
    clearOverlay();
  }
}

function updateFontSourceUI() {
  const source = selectedFontSource();
  const useServer = source === "server";
  const createMode = selectedMode() === "create";
  serverFontField.classList.toggle("hidden", !useServer);
  fontUploadField.classList.toggle("hidden", useServer);

  if (!createMode) {
    serverFontSelect.required = false;
    fontFileInput.required = false;
    return;
  }

  serverFontSelect.required = useServer;
  fontFileInput.required = !useServer;
}

function applyOptions(payload) {
  const serverFonts = toArray(payload?.server_fonts);
  const availableLanguages = toArray(
    payload?.available_languages,
    ["auto", "en", "vi"],
  );

  createLanguageSelect.innerHTML = availableLanguages
    .map((language) => {
      const value = toTrimmedString(language).toLowerCase();
      if (!value) {
        return "";
      }
      const label = languageLabels[value] || value.toUpperCase();
      return `<option value="${value}">${label}</option>`;
    })
    .join("");

  const defaultLanguage = toTrimmedString(
    payload?.default_language,
    "auto",
  ).toLowerCase();
  createLanguageSelect.value = defaultLanguage || "auto";

  if (serverFonts.length === 0) {
    serverFontSelect.innerHTML = "<option value=''>No server fonts available</option>";
    serverFontHelpNode.textContent = "No fonts found in the configured server fonts directory.";
  } else {
    serverFontSelect.innerHTML = serverFonts
      .map((font, index) => {
        const id = toTrimmedString(font?.id);
        const label = toTrimmedString(font?.label) || id || `Font ${index + 1}`;
        if (!id) {
          return "";
        }
        return `<option value="${id}">${label}</option>`;
      })
      .join("");
    serverFontHelpNode.textContent = `Loaded ${serverFonts.length} server fonts.`;
  }

  const defaults = payload?.defaults && typeof payload.defaults === "object"
    ? payload.defaults
    : {};
  Object.entries(numericFieldMap).forEach(([key, node]) => {
    const raw = defaults[key];
    if (!node) {
      return;
    }
    if (raw !== undefined && raw !== null && raw !== "") {
      node.value = String(raw);
    }
  });

  if (showVerticalLineInput) {
    showVerticalLineInput.checked = Boolean(defaults.show_vertical_line);
  }

  const maxCustomTextLength = Number(payload?.max_custom_text_length || 0);
  if (maxCustomTextLength > 0) {
    customTextInput.maxLength = maxCustomTextLength;
  }
}

function appendNumericOverrides(formData) {
  Object.entries(numericFieldMap).forEach(([key, node]) => {
    if (!node) {
      return;
    }
    const value = String(node.value || "").trim();
    if (!value) {
      return;
    }
    formData.append(key, value);
  });
}

function appendBooleanOverrides(formData) {
  if (showVerticalLineInput?.checked) {
    formData.append("show_vertical_line", "true");
  }
}

async function loadSheetOptions() {
  try {
    const response = await fetch(`${apiBaseUrl}/sheet/options`);
    const payload = await parseResponsePayload(response);
    if (!response.ok) {
      const detail = payload?.detail || "Unable to load sheet options";
      throw new Error(String(detail));
    }
    applyOptions(payload);
  } catch (error) {
    setStatus(
      error instanceof Error
        ? `Sheet options warning: ${error.message}`
        : "Sheet options warning: unknown error",
      "warning",
    );
  }
}

async function submitScoreMode() {
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Select an image file before submitting.", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("include_details", "true");

  submitButton.disabled = true;
  setStatus("Scoring sheet...");
  try {
    const response = await fetch(`${apiBaseUrl}/sheet/score`, {
      method: "POST",
      body: formData,
    });

    const payload = await parseResponsePayload(response);
    if (!response.ok) {
      const detail = payload?.detail || "Request failed";
      throw new Error(String(detail));
    }

    renderScoreResults(payload);
    setStatus("Scoring complete.");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Unexpected error", "error");
  } finally {
    submitButton.disabled = false;
  }
}

async function submitCreateMode() {
  const fontSource = selectedFontSource();
  const formData = new FormData();
  formData.append("language", createLanguageSelect.value || "auto");

  const customText = String(customTextInput.value || "").trim();
  if (customText) {
    formData.append("custom_text", customText);
  }

  const outputBasename = String(outputBasenameInput.value || "").trim();
  if (outputBasename) {
    formData.append("output_basename", outputBasename);
  }

  appendNumericOverrides(formData);
  appendBooleanOverrides(formData);

  if (fontSource === "server") {
    const serverFont = String(serverFontSelect.value || "").trim();
    if (!serverFont) {
      setStatus("Select a server font or switch to upload mode.", "warning");
      return;
    }
    formData.append("server_font", serverFont);
  } else {
    const fontFile = fontFileInput.files?.[0];
    if (!fontFile) {
      setStatus("Upload a .ttf or .otf font file before creating.", "warning");
      return;
    }
    formData.append("font_file", fontFile);
  }

  submitButton.disabled = true;
  setStatus("Creating handwriting sheet...");

  try {
    const response = await fetch(`${apiBaseUrl}/sheet/create`, {
      method: "POST",
      body: formData,
    });

    const payload = await parseResponsePayload(response);
    if (!response.ok) {
      const detail = payload?.detail || "Sheet creation failed";
      throw new Error(String(detail));
    }

    renderCreateResult(payload);
    setStatus("Sheet created successfully.");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Unexpected error", "error");
  } finally {
    submitButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) {
    clearPreview();
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

overlayModeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    drawOverlay(lastExtractedBoxes);
  });
});

modeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    resetResults();
    updateModeUI();
  });
});

fontSourceInputs.forEach((input) => {
  input.addEventListener("change", () => {
    updateFontSourceUI();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetResults();

  if (selectedMode() === "create") {
    await submitCreateMode();
    return;
  }
  await submitScoreMode();
});

updateModeUI();
updateFontSourceUI();
loadSheetOptions();
