/* app.js 
  Fixes included:
  - Removed tf.callbacks.earlyStopping restoreBestWeights (not implemented in TFJS) -> replaced with manual early stopping + restore
  - Ensures fitCallbacks is defined before model.fit
  - Sigmoid gate feature importance no longer builds/disposes a model that reuses base layers (prevents "layer already disposed")
  - Feature importance chart rendering fixed (won't be blank)
*/

// ------------------------------
// DOM helpers
// ------------------------------
const $ = (id) => document.getElementById(id);

const statusEl = $("status");
function setStatus(msg) {
  if (statusEl) statusEl.textContent = `Status: ${msg}`;
  console.log(`[STATUS] ${msg}`);
}
function alertUser(msg) {
  console.error(msg);
  window.alert(msg);
}
function setButtonsDisabled(disabled) {
  const ids = [
    "btnInspect",
    "btnTrain",
    "btnEvaluate",
    "btnPredict",
    "btnExportModel",
    "btnDownloadSubmission",
    "btnDownloadProbabilities",
  ];
  for (const id of ids) {
    const el = $(id);
    if (el) el.disabled = disabled;
  }
}
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ------------------------------
// Schema (swap here for other datasets)
// ------------------------------
const SCHEMA = {
  target: "Survived",
  identifier: "PassengerId",
  features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  categorical: ["Pclass", "Sex", "Embarked"],
  numeric: ["Age", "Fare", "SibSp", "Parch"],
};

// ------------------------------
// Global state
// ------------------------------
let trainRows = null;
let testRows = null;
let model = null;

let preprocessState = null;
let valState = null;
let testPredState = null;

let currentThreshold = 0.5;

// ------------------------------
// Robust CSV parsing
// ------------------------------
function parseCSVRobust(text) {
  const input = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  let i = 0;
  let rowIndex = 0;

  function pushField() {
    row.push(field);
    field = "";
  }
  function pushRow() {
    if (!(row.length === 1 && row[0] === "")) rows.push(row);
    row = [];
    rowIndex++;
  }

  while (i < input.length) {
    const c = input[i];

    if (inQuotes) {
      if (c === '"') {
        const next = input[i + 1];
        if (next === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += c;
      i += 1;
      continue;
    }

    // not in quotes
    if (c === '"') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (c === ",") {
      pushField();
      i += 1;
      continue;
    }
    if (c === "\n") {
      pushField();
      pushRow();
      i += 1;
      continue;
    }
    field += c;
    i += 1;
  }

  if (inQuotes) {
    throw new Error(
      `CSV parse error: reached end-of-file with an unclosed quote (approx row ${rowIndex + 1}).`
    );
  }

  pushField();
  if (row.length > 0) pushRow();
  if (rows.length === 0) throw new Error("CSV parse error: no rows found.");

  return rows;
}

function rowsToObjects(matrix) {
  const header = matrix[0].map((h) => h.trim());
  const out = [];

  for (let r = 1; r < matrix.length; r++) {
    const row = matrix[r];
    if (row.length !== header.length) {
      const preview = row.slice(0, 10).join(" | ");
      throw new Error(
        `CSV shape error at data row ${r + 1}: expected ${header.length} columns, got ${row.length}.\n` +
          `Row preview: ${preview}\n` +
          `Tip: check for malformed quotes or stray delimiters in that row.`
      );
    }
    const obj = {};
    for (let c = 0; c < header.length; c++) obj[header[c]] = row[c];
    out.push(obj);
  }
  return { header, rows: out };
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result ?? ""));
    fr.onerror = () => reject(fr.error || new Error("Failed to read file."));
    fr.readAsText(file);
  });
}

async function loadCSVFromFileInput(fileInputEl) {
  const f = fileInputEl?.files?.[0];
  if (!f) return null;

  const text = await readFileAsText(f);
  try {
    const matrix = parseCSVRobust(text);
    const { header, rows } = rowsToObjects(matrix);
    return { header, rows };
  } catch (err) {
    alertUser(`Failed to parse CSV "${f.name}".\n\n${err.message}`);
    return null;
  }
}

// ------------------------------
// Inspection rendering
// ------------------------------
function renderPreviewTable(rows, containerId, limit = 10) {
  const container = $(containerId);
  if (!container) return;

  if (!rows || rows.length === 0) {
    container.innerHTML =
      `<div style="padding:10px;color:rgba(231,236,255,.7);font-size:12px">No data.</div>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows.slice(0, limit).map((r) => {
    const tds = cols.map((c) => `<td>${escapeHtml(r[c] ?? "")}</td>`).join("");
    return `<tr>${tds}</tr>`;
  }).join("");

  container.innerHTML = `
    <table>
      <thead><tr>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderMissingTable(rows, containerId) {
  const container = $(containerId);
  if (!container) return;

  if (!rows || rows.length === 0) {
    container.innerHTML =
      `<div style="padding:10px;color:rgba(231,236,255,.7);font-size:12px">No data.</div>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  const n = rows.length;

  const stats = cols.map((c) => {
    let missing = 0;
    for (const r of rows) {
      const v = r[c];
      if (v == null) missing++;
      else {
        const s = String(v).trim().toLowerCase();
        if (s === "" || s === "na" || s === "nan" || s === "null") missing++;
      }
    }
    return { col: c, missing, pct: (missing / n) * 100 };
  }).sort((a, b) => b.pct - a.pct);

  const body = stats.map((s) => `
    <tr>
      <td>${escapeHtml(s.col)}</td>
      <td>${s.missing}</td>
      <td>${s.pct.toFixed(2)}%</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <table>
      <thead><tr><th>Column</th><th>Missing</th><th>Missing %</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function setShapeText() {
  const trainShapeEl = $("trainShape");
  const testShapeEl = $("testShape");
  if (trainShapeEl) trainShapeEl.textContent =
    trainRows ? `${trainRows.length} × ${Object.keys(trainRows[0] || {}).length}` : "—";
  if (testShapeEl) testShapeEl.textContent =
    testRows ? `${testRows.length} × ${Object.keys(testRows[0] || {}).length}` : "—";
}

// ------------------------------
// Small helpers
// ------------------------------
function toNum(x) {
  if (x == null) return NaN;
  const s = String(x).trim();
  if (s === "") return NaN;
  const v = Number(s);
  return Number.isFinite(v) ? v : NaN;
}
function normalizeCat(x) {
  if (x == null) return "";
  return String(x).trim();
}

// tfjs-vis barchart helper (stable across tfjs-vis versions)
// Expects labels[] and values[] same length.
// Renders as {values: [{x, y}, ...]}
function renderBarChartTFVis(containerEl, title, xLabel, yLabel, labels, values) {
  if (!containerEl) return;

  // Guard against empty input (avoid blank/axes-only charts)
  if (!labels?.length || !values?.length || labels.length !== values.length) {
    containerEl.innerHTML = `
      <div style="color:rgba(231,236,255,.7);font-size:12px;padding:8px">
        Chart unavailable (no data).
      </div>`;
    return;
  }

  const data = {
    values: labels.map((x, i) => ({ x, y: Number(values[i]) })),
  };

  containerEl.innerHTML = "";
  tfvis.render.barchart(containerEl, data, {
    title,
    xLabel,
    yLabel,
    height: 260,
  });
}

// Charts: survival rates
function computeSurvivalRateByGroup(rows, groupCol) {
  const map = new Map();
  for (const r of rows) {
    const g = normalizeCat(r[groupCol]);
    const y = toNum(r[SCHEMA.target]);
    if (!Number.isFinite(y)) continue;

    if (!map.has(g)) map.set(g, { group: g, sum: 0, n: 0 });
    const obj = map.get(g);
    obj.sum += y;
    obj.n += 1;
  }
  return Array.from(map.values())
    .filter((d) => d.n > 0)
    .map((d) => ({ group: d.group, rate: d.sum / d.n, count: d.n }))
    .sort((a, b) => (a.group > b.group ? 1 : -1));
}

function renderBarChartRate(containerEl, title, data) {
  const values = data.map((d) => ({
    Group: d.group,
    SurvivalRate: Number(d.rate.toFixed(4)),
  }));

  containerEl.innerHTML = "";
  tfvis.render.barchart(
    containerEl,
    values,
    { xLabel: "Group", yLabel: "Survival Rate", title, height: 260 }
  );
}

// ------------------------------
// Preprocessing
// ------------------------------
function median(nums) {
  const a = nums.filter((x) => Number.isFinite(x)).slice().sort((x, y) => x - y);
  if (a.length === 0) return NaN;
  const mid = Math.floor(a.length / 2);
  return a.length % 2 === 0 ? (a[mid - 1] + a[mid]) / 2 : a[mid];
}
function mode(strings) {
  const counts = new Map();
  for (const s0 of strings) {
    const s = normalizeCat(s0);
    if (s === "") continue;
    counts.set(s, (counts.get(s) || 0) + 1);
  }
  let best = "";
  let bestN = -1;
  for (const [k, n] of counts.entries()) {
    if (n > bestN) {
      bestN = n;
      best = k;
    }
  }
  return best;
}
function stdStats(nums) {
  const a = nums.filter((x) => Number.isFinite(x));
  if (a.length === 0) return { mean: 0, std: 1 };
  const mean = a.reduce((p, c) => p + c, 0) / a.length;
  const varr = a.reduce((p, c) => p + (c - mean) ** 2, 0) / a.length;
  const std = Math.sqrt(varr);
  return { mean, std: std > 1e-8 ? std : 1 };
}
function buildCategoryMaps(rows, col, knownCategories = null) {
  const cats = new Set();
  for (const r of rows) {
    const v = normalizeCat(r[col]);
    if (v !== "") cats.add(v);
  }
  const list = (knownCategories ? knownCategories : Array.from(cats)).slice().sort();
  const toIndex = new Map(list.map((c, i) => [c, i]));
  return { categories: list, toIndex };
}
function oneHot(index, depth) {
  const arr = new Array(depth).fill(0);
  if (index >= 0 && index < depth) arr[index] = 1;
  return arr;
}

function preprocess(rows, opts) {
  if (!rows || rows.length === 0) throw new Error("No training rows to preprocess.");

  const requiredCols = [SCHEMA.target, SCHEMA.identifier, ...SCHEMA.features];
  const cols = new Set(Object.keys(rows[0] || {}));
  const missingCols = requiredCols.filter((c) => !cols.has(c));
  if (missingCols.length > 0) {
    throw new Error(
      `train.csv is missing required columns: ${missingCols.join(", ")}.\n` +
      `Make sure you used the Kaggle Titanic train.csv file.`
    );
  }

  const ages = rows.map((r) => toNum(r.Age));
  const fares = rows.map((r) => toNum(r.Fare));
  const embarkedVals = rows.map((r) => normalizeCat(r.Embarked));

  const ageMedian = median(ages);
  const embarkedMode = mode(embarkedVals);

  const sexMap = buildCategoryMaps(rows, "Sex");
  const pclassMap = buildCategoryMaps(rows, "Pclass");
  const embarkedMap = buildCategoryMaps(rows, "Embarked", ["C", "Q", "S"]);

  const ageImputed = ages.map((a) => (Number.isFinite(a) ? a : ageMedian));
  const fareImputed = fares.map((f) => (Number.isFinite(f) ? f : 0));
  const ageStats = stdStats(ageImputed);
  const fareStats = stdStats(fareImputed);

  const featureNames = [];
  for (const c of pclassMap.categories) featureNames.push(`Pclass_${c}`);
  for (const c of sexMap.categories) featureNames.push(`Sex_${c}`);
  for (const c of embarkedMap.categories) featureNames.push(`Embarked_${c}`);
  featureNames.push("Age_z");
  featureNames.push("Fare_z");
  featureNames.push("SibSp");
  featureNames.push("Parch");
  if (opts.useFamilySize) featureNames.push("FamilySize");
  if (opts.useIsAlone) featureNames.push("IsAlone");

  const X = [];
  const y = [];

  for (const r of rows) {
    const yy = toNum(r[SCHEMA.target]);
    if (!Number.isFinite(yy)) continue;

    const pclass = normalizeCat(r.Pclass);
    const sex = normalizeCat(r.Sex);
    const embarked = normalizeCat(r.Embarked) || embarkedMode;

    const ageRaw = toNum(r.Age);
    const fareRaw = toNum(r.Fare);

    const age = Number.isFinite(ageRaw) ? ageRaw : ageMedian;
    const fare = Number.isFinite(fareRaw) ? fareRaw : 0;

    const sibsp = Number.isFinite(toNum(r.SibSp)) ? toNum(r.SibSp) : 0;
    const parch = Number.isFinite(toNum(r.Parch)) ? toNum(r.Parch) : 0;

    const familySize = sibsp + parch + 1;
    const isAlone = familySize === 1 ? 1 : 0;

    const pIdx = pclassMap.toIndex.has(pclass) ? pclassMap.toIndex.get(pclass) : -1;
    const sIdx = sexMap.toIndex.has(sex) ? sexMap.toIndex.get(sex) : -1;
    const eIdx = embarkedMap.toIndex.has(embarked) ? embarkedMap.toIndex.get(embarked) : -1;

    const vec = [];
    vec.push(...oneHot(pIdx, pclassMap.categories.length));
    vec.push(...oneHot(sIdx, sexMap.categories.length));
    vec.push(...oneHot(eIdx, embarkedMap.categories.length));

    vec.push((age - ageStats.mean) / ageStats.std);
    vec.push((fare - fareStats.mean) / fareStats.std);
    vec.push(sibsp);
    vec.push(parch);

    if (opts.useFamilySize) vec.push(familySize);
    if (opts.useIsAlone) vec.push(isAlone);

    X.push(vec);
    y.push([yy]);
  }

  if (X.length === 0) throw new Error("After preprocessing, no rows remained.");

  console.log("=== PREPROCESSING SUMMARY ===");
  console.log("Final feature list:", featureNames);
  console.log("X shape:", [X.length, X[0].length], "y shape:", [y.length, 1]);

  return {
    XTensor: tf.tensor2d(X),
    yTensor: tf.tensor2d(y),
    featureNames,
    stats: { ageMedian, embarkedMode, ageStats, fareStats },
    maps: { pclassMap, sexMap, embarkedMap },
    opts,
  };
}

function preprocessTest(rows, prep) {
  if (!rows || rows.length === 0) throw new Error("No test rows to preprocess.");

  const requiredCols = [SCHEMA.identifier, ...SCHEMA.features];
  const cols = new Set(Object.keys(rows[0] || {}));
  const missingCols = requiredCols.filter((c) => !cols.has(c));
  if (missingCols.length > 0) {
    throw new Error(
      `test.csv is missing required columns: ${missingCols.join(", ")}.`
    );
  }

  const { stats, maps, opts } = prep;
  const X = [];
  const passengerIds = [];

  for (const r of rows) {
    passengerIds.push(normalizeCat(r[SCHEMA.identifier]));

    const pclass = normalizeCat(r.Pclass);
    const sex = normalizeCat(r.Sex);
    const embarked = normalizeCat(r.Embarked) || stats.embarkedMode;

    const ageRaw = toNum(r.Age);
    const fareRaw = toNum(r.Fare);

    const age = Number.isFinite(ageRaw) ? ageRaw : stats.ageMedian;
    const fare = Number.isFinite(fareRaw) ? fareRaw : 0;

    const sibsp = Number.isFinite(toNum(r.SibSp)) ? toNum(r.SibSp) : 0;
    const parch = Number.isFinite(toNum(r.Parch)) ? toNum(r.Parch) : 0;

    const familySize = sibsp + parch + 1;
    const isAlone = familySize === 1 ? 1 : 0;

    const pIdx = maps.pclassMap.toIndex.has(pclass) ? maps.pclassMap.toIndex.get(pclass) : -1;
    const sIdx = maps.sexMap.toIndex.has(sex) ? maps.sexMap.toIndex.get(sex) : -1;
    const eIdx = maps.embarkedMap.toIndex.has(embarked) ? maps.embarkedMap.toIndex.get(embarked) : -1;

    const vec = [];
    vec.push(...oneHot(pIdx, maps.pclassMap.categories.length));
    vec.push(...oneHot(sIdx, maps.sexMap.categories.length));
    vec.push(...oneHot(eIdx, maps.embarkedMap.categories.length));

    vec.push((age - stats.ageStats.mean) / stats.ageStats.std);
    vec.push((fare - stats.fareStats.mean) / stats.fareStats.std);
    vec.push(sibsp);
    vec.push(parch);

    if (opts.useFamilySize) vec.push(familySize);
    if (opts.useIsAlone) vec.push(isAlone);

    X.push(vec);
  }

  return { XTensor: tf.tensor2d(X), passengerIds };
}

// ------------------------------
// Stratified split
// ------------------------------
function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffleInPlace(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
function stratifiedSplit(X, y, valFrac = 0.2, seed = 42) {
  const n = y.shape[0];
  const yArr = y.dataSync();

  const idx0 = [];
  const idx1 = [];
  for (let i = 0; i < n; i++) (yArr[i] >= 0.5 ? idx1 : idx0).push(i);

  const rng = mulberry32(seed);
  shuffleInPlace(idx0, rng);
  shuffleInPlace(idx1, rng);

  const nVal0 = Math.floor(idx0.length * valFrac);
  const nVal1 = Math.floor(idx1.length * valFrac);

  const valIdx = idx0.slice(0, nVal0).concat(idx1.slice(0, nVal1));
  const trainIdx = idx0.slice(nVal0).concat(idx1.slice(nVal1));

  shuffleInPlace(trainIdx, rng);
  shuffleInPlace(valIdx, rng);

  return {
    Xtrain: tf.gather(X, trainIdx),
    ytrain: tf.gather(y, trainIdx),
    Xval: tf.gather(X, valIdx),
    yval: tf.gather(y, valIdx),
  };
}

// ------------------------------
// Model
// ------------------------------
function buildModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  m.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });
  console.log("=== MODEL SUMMARY ===");
  m.summary();
  return m;
}

// ------------------------------
// Manual early stopping + restore best weights
// ------------------------------
function makeManualEarlyStopping(model, { monitor = "val_loss", patience = 5, minDelta = 0 } = {}) {
  let best = Infinity;
  let wait = 0;
  let bestWeights = null;

  function disposeWeights(ws) {
    if (!ws) return;
    ws.forEach((t) => t.dispose());
  }

  return {
    onEpochEnd: async (epoch, logs) => {
      const current = logs?.[monitor];
      if (typeof current !== "number" || !Number.isFinite(current)) return;

      if (current < best - minDelta) {
        best = current;
        wait = 0;
        disposeWeights(bestWeights);
        bestWeights = model.getWeights().map((w) => w.clone());
      } else {
        wait += 1;
        if (wait > patience) model.stopTraining = true;
      }
    },
    onTrainEnd: async () => {
      if (bestWeights) {
        model.setWeights(bestWeights);
        disposeWeights(bestWeights);
        bestWeights = null;
      }
    },
  };
}

// ------------------------------
// Metrics
// ------------------------------
function confusionFromProbs(yTrue, probs, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const yt = yTrue[i] >= 0.5 ? 1 : 0;
    const yp = probs[i] >= threshold ? 1 : 0;
    if (yt === 1 && yp === 1) tp++;
    else if (yt === 0 && yp === 1) fp++;
    else if (yt === 0 && yp === 0) tn++;
    else fn++;
  }
  return { tp, fp, tn, fn };
}
function precisionRecallF1(cm) {
  const { tp, fp, fn, tn } = cm;
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  const accuracy = (tp + tn) / Math.max(1, tp + tn + fp + fn);
  return { accuracy, precision, recall, f1 };
}
function rocCurve(yTrue, probs, points = 151) {
  const rocPoints = [];
  for (let i = 0; i < points; i++) {
    const thr = i / (points - 1);
    const cm = confusionFromProbs(yTrue, probs, thr);
    const tpr = (cm.tp + cm.fn) === 0 ? 0 : cm.tp / (cm.tp + cm.fn);
    const fpr = (cm.fp + cm.tn) === 0 ? 0 : cm.fp / (cm.fp + cm.tn);
    rocPoints.push({ thr, fpr, tpr });
  }
  rocPoints.sort((a, b) => a.fpr - b.fpr);

  let auc = 0;
  for (let i = 1; i < rocPoints.length; i++) {
    const x1 = rocPoints[i - 1].fpr;
    const x2 = rocPoints[i].fpr;
    const y1 = rocPoints[i - 1].tpr;
    const y2 = rocPoints[i].tpr;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }
  return { rocPoints, auc: Math.max(0, Math.min(1, auc)) };
}

// ------------------------------
// UI: tables + plots
// ------------------------------
function renderEvalTable(metrics) {
  const wrap = $("evalTableWrap");
  if (!wrap) return;
  wrap.innerHTML = `
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Accuracy</td><td>${metrics.accuracy.toFixed(4)}</td></tr>
        <tr><td>Precision</td><td>${metrics.precision.toFixed(4)}</td></tr>
        <tr><td>Recall</td><td>${metrics.recall.toFixed(4)}</td></tr>
        <tr><td>F1-score</td><td>${metrics.f1.toFixed(4)}</td></tr>
        <tr><td>ROC-AUC</td><td>${metrics.auc.toFixed(4)}</td></tr>
      </tbody>
    </table>
  `;
}
function renderConfusionMatrix(cm) {
  const wrap = $("confusionMatrixWrap");
  if (!wrap) return;
  wrap.innerHTML = `
    <table>
      <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
      <tbody>
        <tr><th>Actual 0</th><td>${cm.tn}</td><td>${cm.fp}</td></tr>
        <tr><th>Actual 1</th><td>${cm.fn}</td><td>${cm.tp}</td></tr>
      </tbody>
    </table>
  `;
}
function renderROC(rocPoints) {
  const el = $("visROC");
  if (!el) return;
  el.innerHTML = "";
  const values = rocPoints.map((p) => ({ x: p.fpr, y: p.tpr }));
  tfvis.render.linechart(
    el,
    { values, series: ["ROC"] },
    { xLabel: "False Positive Rate", yLabel: "True Positive Rate", title: "ROC Curve (Validation)", height: 280 }
  );
}

// ------------------------------
// Sigmoid Gate Feature Importance (SAFE VERSION)
// ------------------------------
/*
  IMPORTANT FIX:
  We do NOT build a tf.Model that reuses base model layers (that can cause "layer already disposed"
  if the derived model is disposed).
  Instead we optimize a gate vector purely with tensors and model.predict().

  This is still a heuristic, NOT SHAP.
*/
async function computeSigmoidGateImportanceTensorOnly(prep, Xval, yval) {
  if (!model) throw new Error("Train the base model before computing sigmoid gate importance.");

  const d = prep.featureNames.length;

  // Gate logits (trainable). gate = sigmoid(logits) in [0,1]
  const gateLogits = tf.variable(tf.zeros([d]), true, "gate_logits");

  const optimizer = tf.train.adam(0.05);
  const yTrue = yval; // tensor [N,1]

  // Small number of steps/epochs for speed
  const steps = 120;
  const batchSize = Math.min(256, Xval.shape[0]);

  // Helper: sample random batch indices
  function randomBatchIdx(n, k) {
    const idx = new Array(k);
    for (let i = 0; i < k; i++) idx[i] = Math.floor(Math.random() * n);
    return idx;
  }

  for (let step = 0; step < steps; step++) {
    tf.tidy(() => {
      const idx = randomBatchIdx(Xval.shape[0], batchSize);
      const xb = tf.gather(Xval, idx);
      const yb = tf.gather(yTrue, idx);

      const lossFn = () => {
        const gate = tf.sigmoid(gateLogits);     // [d]
        const xg = xb.mul(gate);                 // [N,d] (broadcast)
        const preds = model.predict(xg);         // [N,1] probabilities
        const loss = tf.losses.logLoss(yb, preds); // binary cross-entropy
        // Optional tiny regularizer to avoid all-gates-to-0 solutions:
        const reg = gate.mean().mul(1e-4);
        return loss.add(reg);
      };

      optimizer.minimize(lossFn, /*returnCost=*/false, [gateLogits]);
    });
  }

  // Extract final gate values
  const gateValues = await tf.tidy(async () => {
    const g = tf.sigmoid(gateLogits);
    const arr = Array.from(await g.data());
    return arr;
  });

  // Clean up
  gateLogits.dispose();

  // Normalize to sum=1 for "relative importance"
  const sum = gateValues.reduce((p, c) => p + c, 0) || 1;
  return gateValues.map((v) => v / sum);
}

function renderFeatureImportance(featureNames, scores) {
  const el = $("visFeatureImportance");
  if (!el) return;

  if (!featureNames?.length || !scores?.length || featureNames.length !== scores.length) {
    el.innerHTML = `
      <div style="color:rgba(231,236,255,.7);font-size:12px;padding:8px">
        Feature importance unavailable (no data).
      </div>`;
    return;
  }

  const pairs = featureNames.map((f, i) => ({ f, s: Number(scores[i]) }))
                            .sort((a, b) => b.s - a.s);

  renderBarChartTFVis(
    el,
    "Relative Feature Importance (Sigmoid Gate)",
    "Feature",
    "Relative importance (normalized)",
    pairs.map(p => p.f),
    pairs.map(p => p.s)
  );
}

// ------------------------------
// Downloads
// ------------------------------
function downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function toCSVLine(values) {
  return values.map((v) => {
    const s = String(v ?? "");
    const needsQuote = /[",\n]/.test(s);
    const escaped = s.replaceAll('"', '""');
    return needsQuote ? `"${escaped}"` : escaped;
  }).join(",");
}

// ------------------------------
// Threshold slider dynamic updates
// ------------------------------
const thresholdSlider = $("thresholdSlider");
if (thresholdSlider) {
  thresholdSlider.addEventListener("input", () => {
    currentThreshold = Number(thresholdSlider.value);
    $("thresholdValue").textContent = currentThreshold.toFixed(2);

    if (valState?.yTrueArr && valState?.probsArr && valState?.auc != null) {
      const cm = confusionFromProbs(valState.yTrueArr, valState.probsArr, currentThreshold);
      const prf = precisionRecallF1(cm);
      renderConfusionMatrix(cm);
      renderEvalTable({ ...prf, auc: valState.auc });
    }
  });
}

// ------------------------------
// Buttons
// ------------------------------
const btnDownloadSubmission = $("btnDownloadSubmission");
const btnDownloadProbabilities = $("btnDownloadProbabilities");
if (btnDownloadSubmission) btnDownloadSubmission.disabled = true;
if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = true;

$("btnInspect")?.addEventListener("click", async () => {
  try {
    setButtonsDisabled(true);
    setStatus("loading CSV files...");

    const trainData = await loadCSVFromFileInput($("trainFile"));
    const testData = await loadCSVFromFileInput($("testFile"));
    if (!trainData) {
      setStatus("no train.csv loaded");
      return;
    }

    trainRows = trainData.rows;
    testRows = testData?.rows || null;

    setShapeText();
    renderPreviewTable(trainRows, "previewTableWrap", 10);
    renderMissingTable(trainRows, "missingTableWrap");

    const bySex = computeSurvivalRateByGroup(trainRows, "Sex");
    const byPclass = computeSurvivalRateByGroup(trainRows, "Pclass");

    const visSex = $("visSurvivalBySex");
    const visPclass = $("visSurvivalByPclass");
    if (visSex) {
  renderBarChartTFVis(
    visSex,
    "Survival Rate by Sex (Train)",
    "Sex",
    "Survival Rate",
    bySex.map(d => d.group),
    bySex.map(d => d.rate)
  );
}
    if (visPclass) {
  renderBarChartTFVis(
    visPclass,
    "Survival Rate by Pclass (Train)",
    "Pclass",
    "Survival Rate",
    byPclass.map(d => d.group),
    byPclass.map(d => d.rate)
  );
}

    setStatus(`loaded train.csv${testRows ? " + test.csv" : ""}. ready.`);
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("error during inspection");
  } finally {
    setButtonsDisabled(false);
  }
});

$("btnTrain")?.addEventListener("click", async () => {
  let Xtrain, ytrain, Xval, yval;
  try {
    if (!trainRows) {
      alertUser("Please load train.csv and click Inspect Data first.");
      return;
    }

    setButtonsDisabled(true);
    setStatus("preprocessing train.csv...");

    // Dispose old preprocess tensors
    if (preprocessState?.XTensor) preprocessState.XTensor.dispose();
    if (preprocessState?.yTensor) preprocessState.yTensor.dispose();

    const useFamilySize = $("toggleFamilySize")?.checked ?? true;
    const useIsAlone = $("toggleIsAlone")?.checked ?? true;

    preprocessState = preprocess(trainRows, { useFamilySize, useIsAlone });

    setStatus("splitting into stratified train/val (80/20)...");
    ({ Xtrain, ytrain, Xval, yval } = stratifiedSplit(
      preprocessState.XTensor,
      preprocessState.yTensor,
      0.2,
      42
    ));

    // Replace old model
    if (model) {
      model.dispose();
      model = null;
    }

    setStatus("building model...");
    model = buildModel(preprocessState.featureNames.length);

    // Training charts (fitCallbacks MUST be defined)
    const visTraining = $("visTraining");
    if (visTraining) visTraining.innerHTML = "";
    const fitCallbacks = tfvis.show.fitCallbacks(
      visTraining,
      ["loss", "acc", "val_loss", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    );

    const earlyStop = makeManualEarlyStopping(model, { monitor: "val_loss", patience: 5 });

    setStatus("training (max 50 epochs)...");
    await model.fit(Xtrain, ytrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [Xval, yval],
      callbacks: [earlyStop, fitCallbacks],
      shuffle: true,
    });

    // Save val tensors
    if (valState?.Xval) valState.Xval.dispose();
    if (valState?.yval) valState.yval.dispose();
    valState = { Xval, yval, yTrueArr: null, probsArr: null, auc: null };

    // Reset predictions
    testPredState = null;
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = true;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = true;

    setStatus("training complete. click Evaluate.");
  } catch (err) {
    // Clean up split tensors if training failed before we stored them
    if (Xtrain) Xtrain.dispose();
    if (ytrain) ytrain.dispose();
    if (Xval) Xval.dispose();
    if (yval) yval.dispose();

    alertUser(err.message || String(err));
    setStatus("training error");
  } finally {
    setButtonsDisabled(false);
  }
});

$("btnEvaluate")?.addEventListener("click", async () => {
  try {
    if (!model || !preprocessState || !valState?.Xval) {
      alertUser("Please train the model first (load → inspect → train).");
      return;
    }

    setButtonsDisabled(true);
    setStatus("computing validation predictions + metrics...");

    const probsTensor = model.predict(valState.Xval);
    const probsArr = Array.from(await probsTensor.data());
    probsTensor.dispose();

    const yTrueArr = Array.from(await valState.yval.data());

    const { rocPoints, auc } = rocCurve(yTrueArr, probsArr, 151);
    renderROC(rocPoints);

    const cm = confusionFromProbs(yTrueArr, probsArr, currentThreshold);
    const prf = precisionRecallF1(cm);

    renderEvalTable({ ...prf, auc });
    renderConfusionMatrix(cm);

    valState.yTrueArr = yTrueArr;
    valState.probsArr = probsArr;
    valState.auc = auc;

    // SAFE gate importance (tensor-only)
    setStatus("computing sigmoid-gate feature importance (safe heuristic)...");
    const scores = await computeSigmoidGateImportanceTensorOnly(
      preprocessState,
      valState.Xval,
      valState.yval
    );
    renderFeatureImportance(preprocessState.featureNames, scores);

    setStatus("evaluation complete. adjust threshold slider to update metrics.");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("evaluation error");
  } finally {
    setButtonsDisabled(false);
  }
});

$("btnPredict")?.addEventListener("click", async () => {
  try {
    if (!model || !preprocessState) {
      alertUser("Train the model before predicting.");
      return;
    }
    if (!testRows) {
      alertUser("Please load test.csv (in Data Load) and click Inspect Data.");
      return;
    }

    setButtonsDisabled(true);
    setStatus("preprocessing test.csv...");

    const { XTensor, passengerIds } = preprocessTest(testRows, preprocessState);

    setStatus("running inference on test.csv...");
    const probsTensor = model.predict(XTensor);
    const probsArr = Array.from(await probsTensor.data());
    probsTensor.dispose();
    XTensor.dispose();

    const predsArr = probsArr.map((p) => (p >= currentThreshold ? 1 : 0));
    testPredState = { passengerIds, probsArr, predsArr };

    if (btnDownloadSubmission) btnDownloadSubmission.disabled = false;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = false;

    setStatus(`prediction complete. ready to download CSVs (threshold=${currentThreshold.toFixed(2)}).`);
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("prediction error");
  } finally {
    setButtonsDisabled(false);
  }
});

btnDownloadSubmission?.addEventListener("click", () => {
  try {
    if (!testPredState) return alertUser("Run Predict first.");
    const lines = ["PassengerId,Survived"];
    for (let i = 0; i < testPredState.passengerIds.length; i++) {
      lines.push(toCSVLine([testPredState.passengerIds[i], testPredState.predsArr[i]]));
    }
    downloadTextFile("submission.csv", lines.join("\n"));
  } catch (err) {
    alertUser(err.message || String(err));
  }
});

btnDownloadProbabilities?.addEventListener("click", () => {
  try {
    if (!testPredState) return alertUser("Run Predict first.");
    const lines = ["PassengerId,Probability"];
    for (let i = 0; i < testPredState.passengerIds.length; i++) {
      lines.push(toCSVLine([testPredState.passengerIds[i], testPredState.probsArr[i]]));
    }
    downloadTextFile("probabilities.csv", lines.join("\n"));
  } catch (err) {
    alertUser(err.message || String(err));
  }
});

$("btnExportModel")?.addEventListener("click", async () => {
  try {
    if (!model) return alertUser("Train a model first.");
    setButtonsDisabled(true);
    setStatus("exporting model (downloads://titanic-tfjs)...");
    await model.save("downloads://titanic-tfjs");
    setStatus("model exported (check your downloads).");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("model export error");
  } finally {
    setButtonsDisabled(false);
  }
});

// ------------------------------
// Startup
// ------------------------------
setStatus("idle (load train.csv + test.csv, then Inspect Data)");

// ------------------------------
// LLM SELF-REVIEW SUMMARY
// ------------------------------
/*
  LLM SELF-REVIEW SUMMARY

  (1) CSV parsing avoids comma-in-quote bugs
      - parseCSVRobust() scans the CSV text character-by-character while tracking quoted fields.
      - Commas inside quotes are treated as literal characters, so passenger names with commas do not break parsing.
      - Escaped quotes ("") are supported, and row lengths are validated against the header with clear error messages.

  (2) Preprocessing transforms raw Titanic data
      - Target: Survived (0/1). Identifier PassengerId excluded from training.
      - Imputation: Age -> median, Embarked -> mode, Fare missing -> 0.
      - Feature engineering toggles: FamilySize and IsAlone.
      - One-hot encoding for Pclass/Sex/Embarked; z-score standardization for Age/Fare.
      - Outputs tensors X [N,D], y [N,1] and logs final feature list + shapes.

  (3) Model training and evaluation
      - Model: Dense(16,relu) -> Dense(1,sigmoid); compiled with Adam + binaryCrossentropy + accuracy.
      - 80/20 stratified split; manual early stopping on val_loss with best-weight restore; tfjs-vis fitCallbacks for live charts.

  (4) ROC and thresholding
      - Validation probabilities produce ROC curve and trapezoidal ROC-AUC.
      - Threshold slider controls converting probabilities to class labels and dynamically updates confusion matrix + PR/F1/accuracy.

  (5) Sigmoid gate feature importance
      - Not SHAP. A heuristic: learn per-feature gates g=sigmoid(logits) in [0,1].
      - SAFE implementation: tensor-only optimization loop updates gate logits without creating/disposing a model that reuses layers.
      - Normalized gates are displayed as “Relative Feature Importance (Sigmoid Gate)”.
*/
