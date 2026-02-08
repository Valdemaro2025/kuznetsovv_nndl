/* app.js
  Titanic TFJS (browser-only) shallow binary classifier
  - Robust CSV parser (handles commas inside quotes + missing values)
  - Inspection (preview table, shape, missing %, survival bar charts)
  - Preprocessing (impute, one-hot, standardize, optional features)
  - Model (Dense 16 relu -> Dense 1 sigmoid)
  - Training (80/20 stratified split, manual early stopping + restore best weights, tfjs-vis live charts)
  - Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC; dynamic threshold slider)
  - ROC curve plot + confusion matrix
  - Sigmoid gate "feature importance" heuristic visualization
  - Prediction on test.csv and downloads: submission.csv, probabilities.csv
  - Export trained model (downloads://)

  NOTE ABOUT REUSABILITY:
  To adapt this to another dataset, primarily update the SCHEMA section:
   - target column name
   - identifier column (excluded from training)
   - feature columns and which are categorical vs numeric
   - imputation rules and encoders
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
/*
  If you want to use another dataset:
  - Update SCHEMA.target, SCHEMA.identifier, SCHEMA.features
  - Update categorical lists and numeric lists
  - Update imputation rules in preprocess()
*/
const SCHEMA = {
  target: "Survived",
  identifier: "PassengerId",
  // Features to use (raw)
  features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  categorical: ["Pclass", "Sex", "Embarked"],
  numeric: ["Age", "Fare", "SibSp", "Parch"], // Pclass handled as categorical
};

// ------------------------------
// Global state
// ------------------------------
let trainRows = null;     // array of row-objects
let testRows = null;      // array of row-objects
let model = null;

let preprocessState = null;   // stores encoders + stats used
let valState = null;          // stores validation tensors + probs + labels
let testPredState = null;     // stores predictions for test.csv
let currentThreshold = 0.5;

// ------------------------------
// Robust CSV parsing
// ------------------------------
/*
  Avoids the classic split(',') bug by scanning character-by-character and tracking quote state.
  Supports:
    - fields wrapped in double quotes
    - commas inside quoted fields
    - escaped quotes inside quoted fields ("") => "
    - CRLF or LF newlines
  If parsing fails, we show a clear alert indicating the problematic row/reason.
*/
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
        } else {
          inQuotes = false;
          i += 1;
          continue;
        }
      } else {
        field += c;
        i += 1;
        continue;
      }
    } else {
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
      continue;
    }
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
    container.innerHTML = `<div style="padding:10px;color:rgba(231,236,255,.7);font-size:12px">No data.</div>`;
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
    container.innerHTML = `<div style="padding:10px;color:rgba(231,236,255,.7);font-size:12px">No data.</div>`;
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
    const pct = (missing / n) * 100;
    return { col: c, missing, pct };
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
      <thead>
        <tr><th>Column</th><th>Missing</th><th>Missing %</th></tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function setShapeText() {
  const trainShapeEl = $("trainShape");
  const testShapeEl = $("testShape");
  if (trainShapeEl) trainShapeEl.textContent = trainRows ? `${trainRows.length} × ${Object.keys(trainRows[0] || {}).length}` : "—";
  if (testShapeEl) testShapeEl.textContent = testRows ? `${testRows.length} × ${Object.keys(testRows[0] || {}).length}` : "—";
}

// ------------------------------
// Simple aggregations for charts
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
    Count: d.count,
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

  // Basic column presence checks
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

  // Imputation stats
  const ageMedian = median(ages);
  const embarkedMode = mode(embarkedVals);

  // Category maps (fit on train)
  const sexMap = buildCategoryMaps(rows, "Sex");
  const pclassMap = buildCategoryMaps(rows, "Pclass");
  const embarkedMap = buildCategoryMaps(rows, "Embarked", ["C", "Q", "S"]);

  // Standardization stats (fit on train after imputation)
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
    if (!Number.isFinite(yy)) continue; // skip if target missing

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

    // Leave SibSp/Parch as counts (easy to standardize if you want)
    vec.push(sibsp);
    vec.push(parch);

    if (opts.useFamilySize) vec.push(familySize);
    if (opts.useIsAlone) vec.push(isAlone);

    X.push(vec);
    y.push([yy]);
  }

  if (X.length === 0) throw new Error("After preprocessing, no rows remained (is Survived missing?).");

  console.log("=== PREPROCESSING SUMMARY ===");
  console.log("Age median (impute):", ageMedian);
  console.log("Embarked mode (impute):", embarkedMode);
  console.log("Pclass categories:", pclassMap.categories);
  console.log("Sex categories:", sexMap.categories);
  console.log("Embarked categories:", embarkedMap.categories);
  console.log("Age stats:", ageStats);
  console.log("Fare stats:", fareStats);
  console.log("Final feature list:", featureNames);
  console.log("X shape:", [X.length, X[0].length], "y shape:", [y.length, 1]);

  const XTensor = tf.tensor2d(X);
  const yTensor = tf.tensor2d(y);

  return {
    XTensor,
    yTensor,
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
      `test.csv is missing required columns: ${missingCols.join(", ")}.\n` +
      `Make sure you used the Kaggle Titanic test.csv file.`
    );
  }

  const { stats, maps, opts } = prep;
  const X = [];
  const passengerIds = [];

  for (const r of rows) {
    const pid = normalizeCat(r[SCHEMA.identifier]);
    passengerIds.push(pid);

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

  const XTensor = tf.tensor2d(X);
  console.log("Test X shape:", XTensor.shape);
  return { XTensor, passengerIds };
}

// ------------------------------
// Stratified train/val split (80/20)
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

  const Xtrain = tf.gather(X, trainIdx);
  const ytrain = tf.gather(y, trainIdx);
  const Xval = tf.gather(X, valIdx);
  const yval = tf.gather(y, valIdx);

  return { Xtrain, ytrain, Xval, yval };
}

// ------------------------------
// Model building
// ------------------------------
function buildModel(inputDim) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
  m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  m.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("=== MODEL SUMMARY ===");
  m.summary();
  return m;
}

// ------------------------------
// Manual Early Stopping + Restore Best Weights
// ------------------------------
/*
  Fixes the TFJS issue where restoreBestWeights isn't implemented in tf.callbacks.earlyStopping.
  This callback:
    - monitors val_loss
    - stops after patience epochs without improvement
    - restores best weights at train end
*/
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
// Metrics utilities
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
  auc = Math.max(0, Math.min(1, auc));
  return { rocPoints, auc };
}

// ------------------------------
// UI rendering: eval table + confusion matrix + ROC
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
// Sigmoid Gate Feature Importance (heuristic)
// ------------------------------
/*
  Lightweight interpretability heuristic (NOT SHAP):
    - Freeze base model
    - Learn per-feature gate g in [0,1] (g=sigmoid(w))
    - Feed X' = X * g into the frozen model
    - Optimize gates to preserve predictive performance
*/
class SigmoidGateLayer extends tf.layers.Layer {
  constructor(featureDim, config = {}) {
    super(config);
    this.featureDim = featureDim;
  }
  build() {
    this.w = this.addWeight("gateW", [this.featureDim], "float32", tf.initializers.zeros());
    this.built = true;
  }
  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const gate = tf.sigmoid(this.w.read()); // [d]
    return x.mul(gate); // broadcast to [batch,d]
  }
  getConfig() {
    const base = super.getConfig();
    return { ...base, featureDim: this.featureDim };
  }
  static get className() {
    return "SigmoidGateLayer";
  }
}
tf.serialization.registerClass(SigmoidGateLayer);

async function computeSigmoidGateImportance(prep, Xval, yval) {
  if (!model) throw new Error("Train the base model before computing sigmoid gate importance.");

  const featureDim = prep.featureNames.length;

  // Freeze base model weights (post-hoc interpretability)
  model.trainable = false;
  model.layers.forEach((l) => (l.trainable = false));

  // Build: Input -> SigmoidGateLayer -> Frozen base model
  const input = tf.input({ shape: [featureDim] });
  const gateLayer = new SigmoidGateLayer(featureDim);
  const gated = gateLayer.apply(input);
  const out = model.apply(gated);

  const gateModel = tf.model({ inputs: input, outputs: out });
  gateModel.compile({ optimizer: tf.train.adam(0.05), loss: "binaryCrossentropy" });

  // Train gates briefly (heuristic)
  await gateModel.fit(Xval, yval, {
    epochs: 40,
    batchSize: 64,
    verbose: 0,
  });

  // Read sigmoid(gateW) SAFELY.
  // IMPORTANT: Do NOT dispose layer weight variables directly.
  const gateScores = await tf.tidy(async () => {
    // gateLayer.w is a LayerVariable; .read() gives a Tensor
    const wTensor = gateLayer.w.read();          // Tensor [d]
    const gTensor = tf.sigmoid(wTensor);         // Tensor [d]
    const gArr = Array.from(await gTensor.data()); // JS numbers
    return gArr;
  });

  // Normalize to sum=1 so it behaves like "relative importance"
  const sum = gateScores.reduce((p, c) => p + c, 0) || 1;
  const norm = gateScores.map((v) => v / sum);

  // Cleanup model we created (this disposes internal variables safely)
  gateModel.dispose();

  // Unfreeze base model back to normal
  model.trainable = true;
  model.layers.forEach((l) => (l.trainable = true));

  return norm; // length = featureDim, sums to ~1
}

function renderFeatureImportance(featureNames, scores) {
  const el = $("visFeatureImportance");
  if (!el) return;

  // If no data, show a helpful message instead of silent failure
  if (!featureNames?.length || !scores?.length || featureNames.length !== scores.length) {
    el.innerHTML = `<div style="color:rgba(231,236,255,.7);font-size:12px;padding:8px">
      Feature importance unavailable (mismatched feature list / scores).
    </div>`;
    return;
  }

  el.innerHTML = "";

  const values = featureNames.map((name, i) => ({
    Feature: name,
    Importance: Number(scores[i]),
  }));

  // Sort descending so it looks meaningful
  values.sort((a, b) => b.Importance - a.Importance);

  // tfjs-vis barchart expects numeric y field; keep it as "Importance"
  tfvis.render.barchart(
    el,
    values,
    {
      title: "Relative Feature Importance (Sigmoid Gate)",
      xLabel: "Feature",
      yLabel: "Relative importance (normalized)",
      height: 320,
    }
  );
}
// ------------------------------
// Prediction exports
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
// Threshold slider: dynamic updates
// ------------------------------
const thresholdSlider = $("thresholdSlider");
if (thresholdSlider) {
  thresholdSlider.addEventListener("input", () => {
    currentThreshold = Number(thresholdSlider.value);
    const tv = $("thresholdValue");
    if (tv) tv.textContent = currentThreshold.toFixed(2);

    if (valState?.yTrueArr && valState?.probsArr && valState?.auc != null) {
      const cm = confusionFromProbs(valState.yTrueArr, valState.probsArr, currentThreshold);
      const prf = precisionRecallF1(cm);
      renderConfusionMatrix(cm);
      renderEvalTable({ ...prf, auc: valState.auc });
    }
  });
}

// ------------------------------
// Buttons / event wiring
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
    if (visSex) renderBarChartRate(visSex, "Survival Rate by Sex (Train)", bySex);
    if (visPclass) renderBarChartRate(visPclass, "Survival Rate by Pclass (Train)", byPclass);

    setStatus(`loaded train.csv${testRows ? " + test.csv" : ""}. ready.`);
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("error during inspection");
  } finally {
    setButtonsDisabled(false);
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = !testPredState;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = !testPredState;
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

    // Dispose old preprocessed tensors
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

    // Replace old model if re-training
    if (model) {
      model.dispose();
      model = null;
    }

    setStatus("building model...");
    model = buildModel(preprocessState.featureNames.length);

    // tfjs-vis live charts
    const visTraining = $("visTraining");
    if (visTraining) visTraining.innerHTML = "";

    // IMPORTANT: fitCallbacks MUST be defined before model.fit (fixes "fitCallbacks is not defined")
    const fitCallbacks = tfvis.show.fitCallbacks(
      visTraining || { appendChild: () => {} }, // fallback no-op container
      ["loss", "acc", "val_loss", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    );

    // Manual early stopping (fixes restoreBestWeights popup)
    const earlyStop = makeManualEarlyStopping(model, {
      monitor: "val_loss",
      patience: 5,
      minDelta: 0,
    });

    setStatus("training (max 50 epochs, batch=32)...");
    await model.fit(Xtrain, ytrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [Xval, yval],
      callbacks: [earlyStop, fitCallbacks],
      shuffle: true,
    });

    // Store validation tensors for evaluation
    if (valState?.Xval) valState.Xval.dispose();
    if (valState?.yval) valState.yval.dispose();
    valState = { Xval, yval, yTrueArr: null, probsArr: null, auc: null };

    // Reset test predictions because model changed
    testPredState = null;
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = true;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = true;

    setStatus("training complete. click Evaluate.");
  } catch (err) {
    // Dispose split tensors if failure occurs before storing
    if (Xtrain) Xtrain.dispose();
    if (ytrain) ytrain.dispose();
    if (Xval) Xval.dispose();
    if (yval) yval.dispose();

    alertUser(err.message || String(err));
    setStatus("training error");
  } finally {
    setButtonsDisabled(false);
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = !testPredState;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = !testPredState;
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

    // Render UI (fixes "computed but not shown" issue)
    renderEvalTable({ ...prf, auc });
    renderConfusionMatrix(cm);

    valState.yTrueArr = yTrueArr;
    valState.probsArr = probsArr;
    valState.auc = auc;

    setStatus("learning sigmoid-gate feature importance (heuristic)...");
    const scores = await computeSigmoidGateImportance(preprocessState, valState.Xval, valState.yval);
    renderFeatureImportance(preprocessState.featureNames, scores);

    setStatus("evaluation complete. adjust threshold slider to update metrics.");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("evaluation error");
  } finally {
    setButtonsDisabled(false);
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = !testPredState;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = !testPredState;
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
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = !testPredState;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = !testPredState;
  }
});

btnDownloadSubmission?.addEventListener("click", () => {
  try {
    if (!testPredState) {
      alertUser("Run Predict first.");
      return;
    }
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
    if (!testPredState) {
      alertUser("Run Predict first.");
      return;
    }
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
    if (!model) {
      alertUser("Train a model first.");
      return;
    }
    setButtonsDisabled(true);
    setStatus("exporting model (downloads://titanic-tfjs)...");
    await model.save("downloads://titanic-tfjs");
    setStatus("model exported (check your downloads).");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("model export error");
  } finally {
    setButtonsDisabled(false);
    if (btnDownloadSubmission) btnDownloadSubmission.disabled = !testPredState;
    if (btnDownloadProbabilities) btnDownloadProbabilities.disabled = !testPredState;
  }
});

// ------------------------------
// Startup status
// ------------------------------
setStatus("idle (load train.csv + test.csv, then Inspect Data)");

// ------------------------------
// LLM SELF-REVIEW SUMMARY
// ------------------------------
/*
  LLM SELF-REVIEW SUMMARY

  (1) How CSV parsing avoids comma-in-quote bugs
      - The app does NOT parse CSV using naive split(',').
      - parseCSVRobust() scans the CSV text character-by-character while tracking an “inQuotes” state.
      - When inside quotes, commas are treated as literal characters (so names like "Braund, Mr. Owen Harris"
        do not break column alignment).
      - Escaped quotes inside quoted fields are supported via the standard CSV convention: "" becomes ".
      - After parsing, rowsToObjects() validates every row has the same column count as the header. If not,
        it throws an error with the row number, expected/actual columns, and a row preview.

  (2) How preprocessing transforms the raw Titanic data
      - Target: Survived (0/1).
      - Identifier excluded from training: PassengerId.
      - Raw features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
      - Imputation (fit on train only, reused for test):
          * Age: median.
          * Embarked: mode.
          * Fare: missing -> 0.
      - Feature engineering toggles:
          * FamilySize = SibSp + Parch + 1
          * IsAlone = (FamilySize === 1)
      - Encoding:
          * One-hot encode Pclass, Sex, Embarked (category maps from train).
          * Standardize Age and Fare using train mean/std (z-score).
          * SibSp and Parch included as numeric counts.
      - Output tensors: X [N, D], y [N, 1]; feature list + shapes logged to console.

  (3) How the model is trained and evaluated
      - Model: Dense(16, relu) -> Dense(1, sigmoid).
      - Compile: Adam + binaryCrossentropy + accuracy.
      - Split: 80/20 stratified split to preserve class ratio.
      - Training: up to 50 epochs, batch size 32, manual early stopping on val_loss with patience=5,
        and best-weight restoration implemented by cloning and re-setting model weights.
      - Live training charts: tfjs-vis fitCallbacks.

  (4) How ROC and thresholding work
      - The model outputs probabilities on validation.
      - ROC curve: sweep thresholds 0..1, compute TPR/FPR, plot using tfjs-vis.
      - ROC-AUC: trapezoidal integration over ROC points.
      - Threshold slider updates confusion matrix and precision/recall/F1/accuracy dynamically without
        recomputing probabilities.

  (5) How the sigmoid gate provides feature importance
      - Not SHAP; a lightweight heuristic:
          * Freeze trained model.
          * Learn per-feature gates g=sigmoid(w) in [0,1].
          * Feed gated inputs X' = X * g into the frozen model.
          * Train only gate weights briefly to preserve performance.
      - Gates are normalized to sum=1 and displayed as “Relative Feature Importance (Sigmoid Gate)”.
*/
