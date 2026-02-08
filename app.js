/* app.js
  Titanic TFJS (browser-only) shallow binary classifier
  - Robust CSV parser (handles commas inside quotes + missing values)
  - Inspection (preview table, shape, missing %, survival bar charts)
  - Preprocessing (impute, one-hot, standardize, optional features)
  - Model (Dense 16 relu -> Dense 1 sigmoid)
  - Training (80/20 stratified split, early stopping, tfjs-vis live charts)
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
  statusEl.textContent = `Status: ${msg}`;
  console.log(`[STATUS] ${msg}`);
}

function alertUser(msg) {
  console.error(msg);
  window.alert(msg);
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

// UI threshold state
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
  If parsing fails, we attempt to report row and reason.
*/

function parseCSVRobust(text) {
  // Normalize line endings to \n (but we still handle \r)
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
    // Skip completely empty trailing row
    if (!(row.length === 1 && row[0] === "")) {
      rows.push(row);
    }
    row = [];
    rowIndex++;
  }

  while (i < input.length) {
    const c = input[i];

    if (inQuotes) {
      if (c === '"') {
        const next = input[i + 1];
        if (next === '"') {
          // Escaped quote inside quoted field
          field += '"';
          i += 2;
          continue;
        } else {
          // Closing quote
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
      // Not in quotes
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
      // Regular char
      field += c;
      i += 1;
      continue;
    }
  }

  // End of file
  if (inQuotes) {
    // Unclosed quote
    const msg = `CSV parse error: reached end-of-file with an unclosed quote (approx row ${rowIndex + 1}).`;
    throw new Error(msg);
  }
  // Push last field/row if needed
  pushField();
  if (row.length > 0) pushRow();

  if (rows.length === 0) throw new Error("CSV parse error: no rows found.");

  // Ensure rectangular-ish data (some rows may be short due to trailing commas; we still validate later)
  return rows;
}

function rowsToObjects(matrix) {
  const header = matrix[0].map((h) => h.trim());
  const out = [];

  for (let r = 1; r < matrix.length; r++) {
    const row = matrix[r];

    if (row.length !== header.length) {
      // Helpful error message
      const preview = row.slice(0, 8).join(" | ");
      throw new Error(
        `CSV shape error at data row ${r + 1}: expected ${header.length} columns, got ${row.length}.\n` +
        `Row preview: ${preview}\n` +
        `Tip: check for malformed quotes or stray delimiters.`
      );
    }

    const obj = {};
    for (let c = 0; c < header.length; c++) {
      // Keep raw strings for now; preprocessing will coerce types
      obj[header[c]] = row[c];
    }
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
  const f = fileInputEl.files?.[0];
  if (!f) return null;

  const text = await readFileAsText(f);

  try {
    const matrix = parseCSVRobust(text);
    const { header, rows } = rowsToObjects(matrix);
    return { header, rows };
  } catch (err) {
    // Try to highlight approximate problematic row if possible
    alertUser(`Failed to parse CSV "${f.name}".\n\n${err.message}`);
    return null;
  }
}

// ------------------------------
// Inspection rendering
// ------------------------------
function renderPreviewTable(rows, containerId, limit = 10) {
  const container = $(containerId);
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div style="padding:10px;color:rgba(231,236,255,.7);font-size:12px">No data.</div>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows.slice(0, limit).map((r) => {
    const tds = cols.map((c) => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("");
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
        const s = String(v).trim();
        if (s === "" || s.toLowerCase() === "na" || s.toLowerCase() === "nan") missing++;
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

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setShapeText() {
  $("trainShape").textContent = trainRows ? `${trainRows.length} × ${Object.keys(trainRows[0] || {}).length}` : "—";
  $("testShape").textContent = testRows ? `${testRows.length} × ${Object.keys(testRows[0] || {}).length}` : "—";
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
  // returns [{group, rate, count}]
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
  // tfjs-vis expects array of objects with consistent keys
  const values = data.map((d) => ({
    Group: d.group,
    SurvivalRate: Number(d.rate.toFixed(4)),
    Count: d.count,
  }));

  containerEl.innerHTML = ""; // clear
  tfvis.render.barchart(
    containerEl,
    values,
    {
      xLabel: "Group",
      yLabel: "Survival Rate",
      title,
      height: 260,
    }
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
  for (const s of strings) {
    const k = normalizeCat(s);
    if (k === "") continue;
    counts.set(k, (counts.get(k) || 0) + 1);
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
  // Ensure stable deterministic order
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
  // opts: {useFamilySize, useIsAlone}
  if (!rows || rows.length === 0) throw new Error("No training rows to preprocess.");

  // Extract raw columns
  const ages = rows.map((r) => toNum(r.Age));
  const fares = rows.map((r) => toNum(r.Fare));
  const embarkedVals = rows.map((r) => normalizeCat(r.Embarked));

  // Imputation stats
  const ageMedian = median(ages);
  const embarkedMode = mode(embarkedVals);

  // Category maps (fit on train)
  const sexMap = buildCategoryMaps(rows, "Sex");
  const pclassMap = buildCategoryMaps(rows, "Pclass");      // will be values like "1","2","3"
  const embarkedMap = buildCategoryMaps(rows, "Embarked", ["C", "Q", "S"].filter((x) => x)); // keep common order

  // Standardization stats (fit on train after imputation)
  const ageImputed = ages.map((a) => (Number.isFinite(a) ? a : ageMedian));
  const fareImputed = fares.map((f) => (Number.isFinite(f) ? f : 0)); // Fare sometimes missing; 0 is reasonable fallback
  const ageStats = stdStats(ageImputed);
  const fareStats = stdStats(fareImputed);

  const featureNames = [];

  // Build feature names in the same order we create vectors
  // Categorical one-hot names
  for (const c of pclassMap.categories) featureNames.push(`Pclass_${c}`);
  for (const c of sexMap.categories) featureNames.push(`Sex_${c}`);
  for (const c of embarkedMap.categories) featureNames.push(`Embarked_${c}`);

  // Numeric standardized
  featureNames.push("Age_z");
  featureNames.push("Fare_z");
  featureNames.push("SibSp");
  featureNames.push("Parch");

  if (opts.useFamilySize) featureNames.push("FamilySize");
  if (opts.useIsAlone) featureNames.push("IsAlone");

  // Vectorize rows
  const X = [];
  const y = [];

  for (const r of rows) {
    // Target
    const yy = toNum(r[SCHEMA.target]);
    if (!Number.isFinite(yy)) {
      // If target missing, skip row
      continue;
    }

    // Raw values
    const pclass = normalizeCat(r.Pclass);
    const sex = normalizeCat(r.Sex);
    const embarked = normalizeCat(r.Embarked) || embarkedMode;

    const age = Number.isFinite(toNum(r.Age)) ? toNum(r.Age) : ageMedian;
    const fare = Number.isFinite(toNum(r.Fare)) ? toNum(r.Fare) : 0;

    const sibsp = Number.isFinite(toNum(r.SibSp)) ? toNum(r.SibSp) : 0;
    const parch = Number.isFinite(toNum(r.Parch)) ? toNum(r.Parch) : 0;

    const familySize = sibsp + parch + 1;
    const isAlone = familySize === 1 ? 1 : 0;

    // Categorical one-hots
    const pIdx = pclassMap.toIndex.has(pclass) ? pclassMap.toIndex.get(pclass) : -1;
    const sIdx = sexMap.toIndex.has(sex) ? sexMap.toIndex.get(sex) : -1;
    const eIdx = embarkedMap.toIndex.has(embarked) ? embarkedMap.toIndex.get(embarked) : -1;

    const vec = [];
    vec.push(...oneHot(pIdx, pclassMap.categories.length));
    vec.push(...oneHot(sIdx, sexMap.categories.length));
    vec.push(...oneHot(eIdx, embarkedMap.categories.length));

    // Numeric standardized
    vec.push((age - ageStats.mean) / ageStats.std);
    vec.push((fare - fareStats.mean) / fareStats.std);
    vec.push(sibsp);
    vec.push(parch);

    if (opts.useFamilySize) vec.push(familySize);
    if (opts.useIsAlone) vec.push(isAlone);

    X.push(vec);
    y.push([yy]);
  }

  if (X.length === 0) throw new Error("After preprocessing, no rows remained (target missing?).");

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
    stats: {
      ageMedian,
      embarkedMode,
      ageStats,
      fareStats,
    },
    maps: {
      pclassMap,
      sexMap,
      embarkedMap,
    },
    opts,
  };
}

function preprocessTest(rows, prep) {
  if (!rows || rows.length === 0) throw new Error("No test rows to preprocess.");
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
function stratifiedSplit(X, y, valFrac = 0.2, seed = 42) {
  // X: tf.Tensor2d, y: tf.Tensor2d with shape [n,1]
  // We'll do index-based split preserving class ratio.
  const n = y.shape[0];

  const yArr = y.dataSync(); // length n
  const idx0 = [];
  const idx1 = [];
  for (let i = 0; i < n; i++) {
    (yArr[i] >= 0.5 ? idx1 : idx0).push(i);
  }

  // Deterministic-ish shuffle using a simple seeded RNG
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

function mulberry32(a) {
  return function () {
    let t = a += 0x6D2B79F5;
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
// Metrics utilities
// ------------------------------
function confusionFromProbs(yTrue, probs, threshold) {
  // yTrue: Array<number> of 0/1
  // probs: Array<number> of [0..1]
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
  const accuracy = (tp + tn) / Math.max(1, (tp + tn + fp + fn));
  return { accuracy, precision, recall, f1 };
}

function rocCurve(yTrue, probs, points = 101) {
  // returns { rocPoints: [{fpr,tpr}], auc }
  // We sweep thresholds from 0..1
  const rocPoints = [];
  for (let i = 0; i < points; i++) {
    const thr = i / (points - 1);
    const cm = confusionFromProbs(yTrue, probs, thr);
    const tpr = (cm.tp + cm.fn) === 0 ? 0 : cm.tp / (cm.tp + cm.fn);
    const fpr = (cm.fp + cm.tn) === 0 ? 0 : cm.fp / (cm.fp + cm.tn);
    rocPoints.push({ thr, fpr, tpr });
  }
  // Sort by FPR (monotonic-ish)
  rocPoints.sort((a, b) => a.fpr - b.fpr);

  // Trapezoidal AUC
  let auc = 0;
  for (let i = 1; i < rocPoints.length; i++) {
    const x1 = rocPoints[i - 1].fpr;
    const x2 = rocPoints[i].fpr;
    const y1 = rocPoints[i - 1].tpr;
    const y2 = rocPoints[i].tpr;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }
  // Clamp
  auc = Math.max(0, Math.min(1, auc));
  return { rocPoints, auc };
}

// ------------------------------
// UI rendering: eval table + confusion matrix + ROC
// ------------------------------
function renderEvalTable(metrics) {
  const wrap = $("evalTableWrap");
  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Metric</th><th>Value</th>
        </tr>
      </thead>
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
  // Convention: rows = Actual, cols = Predicted
  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th></th>
          <th>Pred 0</th>
          <th>Pred 1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Actual 0</th>
          <td>${cm.tn}</td>
          <td>${cm.fp}</td>
        </tr>
        <tr>
          <th>Actual 1</th>
          <td>${cm.fn}</td>
          <td>${cm.tp}</td>
        </tr>
      </tbody>
    </table>
  `;
}

function renderROC(rocPoints) {
  const el = $("visROC");
  el.innerHTML = "";
  // tfvis.render.linechart expects series: [{x,y}]
  const values = rocPoints.map((p) => ({ x: p.fpr, y: p.tpr }));
  tfvis.render.linechart(
    el,
    { values, series: ["ROC"] },
    {
      xLabel: "False Positive Rate",
      yLabel: "True Positive Rate",
      title: "ROC Curve (Validation)",
      height: 280,
    }
  );
}

// ------------------------------
// Sigmoid Gate Feature Importance (heuristic)
// ------------------------------
/*
  This is a lightweight interpretability heuristic, NOT SHAP.

  Idea:
    - Freeze the trained model.
    - Learn a per-feature gate g in [0,1] (g = sigmoid(w)).
    - Feed gated inputs X' = X * g into the frozen model.
    - Optimize gates to preserve validation performance.

  The resulting gate values are treated as *relative* feature importance scores.
*/
class SigmoidGateLayer extends tf.layers.Layer {
  constructor(featureDim, config = {}) {
    super(config);
    this.featureDim = featureDim;
  }
  build(inputShape) {
    this.w = this.addWeight(
      "gateW",
      [this.featureDim],
      "float32",
      tf.initializers.zeros()
    );
    this.built = true;
  }
  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const gate = tf.sigmoid(this.w.read()); // shape [d]
    // broadcast to [batch,d]
    return x.mul(gate);
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

  // Freeze base model weights
  model.trainable = false;
  model.layers.forEach((l) => (l.trainable = false));

  // Build gate model: input -> gate -> frozen base model
  const input = tf.input({ shape: [featureDim] });
  const gated = new SigmoidGateLayer(featureDim).apply(input);
  const out = model.apply(gated);
  const gateModel = tf.model({ inputs: input, outputs: out });

  gateModel.compile({
    optimizer: tf.train.adam(0.05),
    loss: "binaryCrossentropy",
  });

  // Quick fit on validation to learn gates (small epochs)
  // (You can also train on train set; using val keeps it post-hoc.)
  await gateModel.fit(Xval, yval, {
    epochs: 40,
    batchSize: 64,
    verbose: 0,
  });

  // Extract gate values
  const gateLayer = gateModel.layers.find((l) => l.getClassName() === "SigmoidGateLayer");
  const gateW = gateLayer.getWeights()[0]; // [d]
  const gateVals = tf.sigmoid(gateW).dataSync(); // Float32Array

  // Normalize to sum=1 for easier reading (optional)
  const sum = gateVals.reduce((p, c) => p + c, 0) || 1;
  const norm = Array.from(gateVals).map((v) => v / sum);

  gateModel.dispose();
  gateW.dispose();

  // Restore model trainable flag (for safety)
  model.trainable = true;
  model.layers.forEach((l) => (l.trainable = true));

  return norm; // array length d, sums to 1
}

function renderFeatureImportance(featureNames, scores) {
  const el = $("visFeatureImportance");
  el.innerHTML = "";

  // Build values for tfvis
  const values = featureNames.map((name, i) => ({
    Feature: name,
    Importance: Number(scores[i].toFixed(6)),
  }));

  // Sort descending for readability
  values.sort((a, b) => b.Importance - a.Importance);

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
  // CSV-safe (quote when needed; escape quotes)
  return values.map((v) => {
    const s = String(v ?? "");
    const needsQuote = /[",\n]/.test(s);
    const escaped = s.replaceAll('"', '""');
    return needsQuote ? `"${escaped}"` : escaped;
  }).join(",");
}

// ------------------------------
// Buttons / event wiring
// ------------------------------
$("btnDownloadSubmission").disabled = true;
$("btnDownloadProbabilities").disabled = true;

$("thresholdSlider").addEventListener("input", () => {
  currentThreshold = Number($("thresholdSlider").value);
  $("thresholdValue").textContent = currentThreshold.toFixed(2);

  // Dynamic metric update if we already evaluated
  if (valState?.yTrueArr && valState?.probsArr && valState?.auc != null) {
    const cm = confusionFromProbs(valState.yTrueArr, valState.probsArr, currentThreshold);
    const prf = precisionRecallF1(cm);

    renderConfusionMatrix(cm);
    renderEvalTable({
      ...prf,
      auc: valState.auc,
    });
  }
});

$("btnInspect").addEventListener("click", async () => {
  try {
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

    // Preview + missing values
    renderPreviewTable(trainRows, "previewTableWrap", 10);
    renderMissingTable(trainRows, "missingTableWrap");

    // Charts: survival rate by Sex and Pclass
    const bySex = computeSurvivalRateByGroup(trainRows, "Sex");
    const byPclass = computeSurvivalRateByGroup(trainRows, "Pclass");

    renderBarChartRate($("visSurvivalBySex"), "Survival Rate by Sex (Train)", bySex);
    renderBarChartRate($("visSurvivalByPclass"), "Survival Rate by Pclass (Train)", byPclass);

    setStatus(`loaded train.csv${testRows ? " + test.csv" : ""}. ready.`);
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("error during inspection");
  }
});

$("btnTrain").addEventListener("click", async () => {
  try {
    if (!trainRows) {
      alertUser("Please load train.csv and click Inspect Data first.");
      return;
    }

    setStatus("preprocessing train.csv...");
    tf.engine().startScope();

    // Read preprocessing toggles
    const useFamilySize = $("toggleFamilySize").checked;
    const useIsAlone = $("toggleIsAlone").checked;

    // Build tensors
    if (preprocessState?.XTensor) {
      // Clean up old tensors if retraining
      preprocessState.XTensor.dispose();
      preprocessState.yTensor.dispose();
    }

    preprocessState = preprocess(trainRows, { useFamilySize, useIsAlone });

    // Split
    setStatus("splitting into stratified train/val (80/20)...");
    const { Xtrain, ytrain, Xval, yval } = stratifiedSplit(
      preprocessState.XTensor, preprocessState.yTensor, 0.2, 42
    );

    // Build model
    setStatus("building model...");
    if (model) {
      model.dispose();
      model = null;
    }
    model = buildModel(preprocessState.featureNames.length);

    // Training UI
    $("visTraining").innerHTML = "";
    const surface = { name: "Training", tab: "Training", styles: { height: "280px" } };
    const container = $("visTraining");

    // Early stopping
    const earlyStop = tf.callbacks.earlyStopping({
      monitor: "val_loss",
      patience: 5,
      restoreBestWeights: true,
    });

    setStatus("training (50 epochs max, batch=32)...");
    const fitCallbacks = tfvis.show.fitCallbacks(
      container,
      ["loss", "acc", "val_loss", "val_acc"],
      { callbacks: ["onEpochEnd"] }
    );

    await model.fit(Xtrain, ytrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [Xval, yval],
      callbacks: [earlyStop, fitCallbacks],
      shuffle: true,
    });

    // Persist validation tensors/state for evaluation
    if (valState?.Xval) {
      valState.Xval.dispose();
      valState.yval.dispose();
    }
    valState = { Xval, yval, yTrueArr: null, probsArr: null, auc: null };

    setStatus("training complete. click Evaluate.");
    tf.engine().endScope();
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("training error");
    tf.engine().endScope();
  }
});

$("btnEvaluate").addEventListener("click", async () => {
  try {
    if (!model || !preprocessState || !valState?.Xval) {
      alertUser("Please train the model first (load → inspect → train).");
      return;
    }

    setStatus("computing validation predictions + metrics...");

    // Get validation probs
    const probsTensor = model.predict(valState.Xval);
    const probsArr = Array.from((await probsTensor.data()));
    probsTensor.dispose();

    const yTrueArr = Array.from((await valState.yval.data()));

    // ROC + AUC
    const { rocPoints, auc } = rocCurve(yTrueArr, probsArr, 151);
    renderROC(rocPoints);

    // Compute metrics at current threshold
    const cm = confusionFromProbs(yTrueArr, probsArr, currentThreshold);
    const prf = precisionRecallF1(cm);

    // IMPORTANT: render the evaluation table (fixes "computed but not shown" issue)
    renderEvalTable({ ...prf, auc });
    renderConfusionMatrix(cm);

    // Save val state for slider dynamic updates
    valState.yTrueArr = yTrueArr;
    valState.probsArr = probsArr;
    valState.auc = auc;

    // Sigmoid gate feature importance
    setStatus("learning sigmoid-gate feature importance (heuristic)...");
    const scores = await computeSigmoidGateImportance(preprocessState, valState.Xval, valState.yval);
    renderFeatureImportance(preprocessState.featureNames, scores);

    setStatus("evaluation complete. adjust threshold slider to update metrics.");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("evaluation error");
  }
});

$("btnPredict").addEventListener("click", async () => {
  try {
    if (!model || !preprocessState) {
      alertUser("Train the model before predicting.");
      return;
    }
    if (!testRows) {
      alertUser("Please load test.csv (in Data Load) and click Inspect Data.");
      return;
    }

    setStatus("preprocessing test.csv...");
    const { XTensor, passengerIds } = preprocessTest(testRows, preprocessState);

    setStatus("running inference on test.csv...");
    const probsTensor = model.predict(XTensor);
    const probsArr = Array.from(await probsTensor.data());

    probsTensor.dispose();
    XTensor.dispose();

    // Apply threshold
    const predsArr = probsArr.map((p) => (p >= currentThreshold ? 1 : 0));

    testPredState = { passengerIds, probsArr, predsArr };

    $("btnDownloadSubmission").disabled = false;
    $("btnDownloadProbabilities").disabled = false;

    setStatus(`prediction complete. ready to download CSVs (threshold=${currentThreshold.toFixed(2)}).`);
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("prediction error");
  }
});

$("btnDownloadSubmission").addEventListener("click", () => {
  try {
    if (!testPredState) {
      alertUser("Run Predict first.");
      return;
    }
    const lines = [];
    lines.push("PassengerId,Survived");
    for (let i = 0; i < testPredState.passengerIds.length; i++) {
      lines.push(toCSVLine([testPredState.passengerIds[i], testPredState.predsArr[i]]));
    }
    downloadTextFile("submission.csv", lines.join("\n"));
  } catch (err) {
    alertUser(err.message || String(err));
  }
});

$("btnDownloadProbabilities").addEventListener("click", () => {
  try {
    if (!testPredState) {
      alertUser("Run Predict first.");
      return;
    }
    const lines = [];
    lines.push("PassengerId,Probability");
    for (let i = 0; i < testPredState.passengerIds.length; i++) {
      lines.push(toCSVLine([testPredState.passengerIds[i], testPredState.probsArr[i]]));
    }
    downloadTextFile("probabilities.csv", lines.join("\n"));
  } catch (err) {
    alertUser(err.message || String(err));
  }
});

$("btnExportModel").addEventListener("click", async () => {
  try {
    if (!model) {
      alertUser("Train a model first.");
      return;
    }
    setStatus("exporting model (downloads://titanic-tfjs)...");
    await model.save("downloads://titanic-tfjs");
    setStatus("model exported (check your downloads).");
  } catch (err) {
    alertUser(err.message || String(err));
    setStatus("model export error");
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

  (1) CSV parsing avoids comma-in-quote bugs
      - This app does NOT use naive split(',').
      - parseCSVRobust() scans the file character-by-character while tracking whether it is inside quotes.
      - Commas inside quoted strings (like passenger names: "Braund, Mr. Owen Harris") are treated as literal characters,
        not delimiters.
      - It also supports escaped quotes ("") and both LF/CRLF newlines.
      - If a row has the wrong number of columns, rowsToObjects() throws a clear error with the row index and a preview.

  (2) Preprocessing transforms raw Titanic data
      - Target: Survived (0/1).
      - Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked; PassengerId is treated as an identifier (excluded).
      - Imputation:
          * Age: median of training Age values.
          * Embarked: mode of training Embarked values.
          * Fare: missing values default to 0 (rare; still standardized).
      - Feature engineering toggles:
          * FamilySize = SibSp + Parch + 1
          * IsAlone = (FamilySize === 1)
      - Encoding:
          * One-hot encode categorical fields: Pclass, Sex, Embarked.
          * Standardize numeric fields: Age and Fare (z-score using training mean/std).
          * SibSp and Parch are kept as numeric counts (not standardized here; easy to change).
      - The final feature list and X/y tensor shapes are logged to the console.

  (3) Model training and evaluation
      - Model: tf.sequential() with Dense(16, relu) -> Dense(1, sigmoid).
      - Compile: optimizer='adam', loss='binaryCrossentropy', metrics=['accuracy'].
      - Data split: 80/20 stratified train/validation split to preserve class ratios.
      - Training: up to 50 epochs, batch size 32, early stopping on val_loss (patience=5, restore best weights).
      - Live training curves: tfjs-vis fitCallbacks render loss/accuracy as training runs.

  (4) ROC and thresholding mechanics
      - Evaluation produces validation probabilities (sigmoid outputs).
      - ROC curve is computed by sweeping thresholds from 0..1, computing TPR/FPR each time,
        then plotting TPR vs FPR with tfjs-vis linechart.
      - ROC-AUC is computed with a trapezoidal approximation over ROC points.
      - A threshold slider (0–1) updates:
          * confusion matrix (TN/FP/FN/TP)
          * Accuracy, Precision, Recall, and F1-score
        dynamically in the UI, without recomputing probabilities.

  (5) Sigmoid gate feature importance
      - This is a lightweight interpretability heuristic (NOT SHAP).
      - After training, the base model is frozen.
      - A small “gate model” is built: input -> SigmoidGateLayer -> frozen base model.
      - SigmoidGateLayer learns a per-feature gate g in [0,1] (g = sigmoid(w)).
      - Gates are trained on the validation set to keep predictive performance while revealing which input
        dimensions the model relies on.
      - The gate values are normalized to sum to 1 and shown as a tfjs-vis bar chart titled
        “Relative Feature Importance (Sigmoid Gate)”.
*/
