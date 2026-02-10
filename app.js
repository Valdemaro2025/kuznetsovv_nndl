// app.js - Titanic Classifier with Focus on Real CSV Files

// ============================================================================
// GLOBAL STATE AND CONFIGURATION
// ============================================================================

let appState = {
    rawTrainData: null,
    rawTestData: null,
    processedTrainData: null,
    processedTestData: null,
    model: null,
    featureStats: {},
    featureNames: [],
    validationData: null,
    testPredictions: null,
    testPassengerIds: null,
    trainingHistory: []
};

const DATA_SCHEMA = {
    targetColumn: 'Survived',
    idColumn: 'PassengerId',
    numericalFeatures: ['Age', 'Fare'],
    rawNumericalFeatures: ['SibSp', 'Parch'],
    categoricalFeatures: ['Pclass', 'Sex', 'Embarked'],
    excludeColumns: ['Name', 'Ticket', 'Cabin']
};

// ============================================================================
// CORE UTILITY FUNCTIONS
// ============================================================================

function updateStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    const icon = type === 'error' ? 'exclamation-triangle' : 
                 type === 'success' ? 'check-circle' : 'info-circle';
    
    element.innerHTML = `<i class="fas fa-${icon}"></i> ${message}`;
    element.className = `status ${type}`;
}

function showAlert(message, type = 'info') {
    alert(`[${type.toUpperCase()}] ${message}`);
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ============================================================================
// CSV FILE PARSING - ROBUST VERSION
// ============================================================================

async function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        if (!file) {
            reject(new Error('No file provided'));
            return;
        }

        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const text = e.target.result;
                
                // Parse CSV with proper handling of quoted fields
                const rows = parseCSV(text);
                
                if (rows.length < 1) {
                    reject(new Error('CSV file is empty or could not be parsed'));
                    return;
                }
                
                console.log(`Parsed ${rows.length} rows from ${file.name}`);
                resolve(rows);
                
            } catch (error) {
                reject(new Error(`CSV parsing error: ${error.message}`));
            }
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    const lines = csvText.split(/\r\n|\n|\r/);
    const result = [];
    
    if (lines.length === 0) return result;
    
    // Get headers from first line
    const headers = parseCSVLine(lines[0]);
    
    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue; // Skip empty lines
        
        const values = parseCSVLine(line);
        const row = {};
        
        headers.forEach((header, idx) => {
            if (idx < values.length) {
                let value = values[idx];
                
                // Convert to appropriate type
                if (value === '' || value === 'NULL' || value === 'null' || value === 'NA' || value === 'N/A') {
                    value = null;
                } else if (!isNaN(value) && value !== '') {
                    // Try to convert to number
                    const num = parseFloat(value);
                    if (!isNaN(num)) {
                        value = num;
                    }
                }
                
                row[header] = value;
            }
        });
        
        result.push(row);
    }
    
    return result;
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                current += '"';
                i++; // Skip next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    
    result.push(current);
    return result.map(val => val.trim());
}

// ============================================================================
// DATA LOADING - PRIORITY ON CSV FILES
// ============================================================================

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
        const useDemoData = document.getElementById('useDemoData').checked;
        
        updateStatus('loadStatus', 'Loading data...', 'info');
        
        // Clear previous data
        appState.rawTrainData = null;
        appState.rawTestData = null;
        
        if (trainFile) {
            // Use uploaded files
            appState.rawTrainData = await parseCSVFile(trainFile);
            
            if (testFile) {
                appState.rawTestData = await parseCSVFile(testFile);
            } else {
                // Split training data if no test file provided
                const testSize = Math.floor(appState.rawTrainData.length * 0.2);
                appState.rawTestData = appState.rawTrainData.slice(-testSize);
                appState.rawTrainData = appState.rawTrainData.slice(0, -testSize);
            }
            
            updateStatus('loadStatus', 
                `✓ Loaded ${appState.rawTrainData.length} training and ${appState.rawTestData.length} test samples from CSV files`, 
                'success');
                
            // Disable demo data checkbox
            document.getElementById('useDemoData').checked = false;
            
        } else if (useDemoData) {
            // Fallback to demo data
            await loadDemoData();
            updateStatus('loadStatus', 
                `✓ Loaded demo data: ${appState.rawTrainData.length} training samples`, 
                'success');
        } else {
            updateStatus('loadStatus', 'Please upload train.csv or check "Use demo data"', 'error');
            return false;
        }
        
        // Validate loaded data
        if (!validateData()) {
            return false;
        }
        
        // Enable training button
        document.getElementById('trainBtn').disabled = false;
        
        // Display data inspection
        inspectData();
        
        return true;
        
    } catch (error) {
        updateStatus('loadStatus', `✗ Failed to load data: ${error.message}`, 'error');
        console.error('Load error:', error);
        return false;
    }
}

function validateData() {
    if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
        updateStatus('loadStatus', 'Training data is empty', 'error');
        return false;
    }
    
    // Check for required columns in training data
    const requiredColumns = [DATA_SCHEMA.targetColumn, DATA_SCHEMA.idColumn];
    const firstRow = appState.rawTrainData[0];
    const missingColumns = [];
    
    for (const col of requiredColumns) {
        if (!(col in firstRow)) {
            missingColumns.push(col);
        }
    }
    
    if (missingColumns.length > 0) {
        updateStatus('loadStatus', 
            `CSV file missing required columns: ${missingColumns.join(', ')}`, 
            'error');
        return false;
    }
    
    // Log data statistics
    console.log('Data validation passed:', {
        trainSamples: appState.rawTrainData.length,
        testSamples: appState.rawTestData ? appState.rawTestData.length : 0,
        columns: Object.keys(firstRow)
    });
    
    return true;
}

async function loadDemoData() {
    // Simple demo data for testing
    appState.rawTrainData = [
        {PassengerId: 1, Survived: 0, Pclass: 3, Name: 'Demo 1', Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Ticket: 'T1', Fare: 7.25, Embarked: 'S'},
        {PassengerId: 2, Survived: 1, Pclass: 1, Name: 'Demo 2', Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Ticket: 'T2', Fare: 71.28, Embarked: 'C'},
        {PassengerId: 3, Survived: 1, Pclass: 3, Name: 'Demo 3', Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Ticket: 'T3', Fare: 7.92, Embarked: 'S'}
    ];
    
    appState.rawTestData = [
        {PassengerId: 11, Pclass: 3, Name: 'Test 1', Sex: 'male', Age: 20, SibSp: 0, Parch: 0, Ticket: 'T11', Fare: 8.05, Embarked: 'S'},
        {PassengerId: 12, Pclass: 1, Name: 'Test 2', Sex: 'female', Age: 58, SibSp: 0, Parch: 0, Ticket: 'T12', Fare: 26.55, Embarked: 'C'}
    ];
    
    return true;
}

// ============================================================================
// DATA INSPECTION - SAFE VERSION
// ============================================================================

function inspectData() {
    if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
        document.getElementById('inspectionContent').innerHTML = 
            '<div class="status error">No data loaded</div>';
        return;
    }
    
    const data = appState.rawTrainData;
    const totalRows = data.length;
    const columns = Object.keys(data[0]);
    
    // Create data preview table
    let previewHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewHTML += '<div class="data-preview"><table>';
    
    // Table header
    previewHTML += '<thead><tr>';
    columns.forEach(col => {
        previewHTML += `<th>${escapeHtml(col)}</th>`;
    });
    previewHTML += '</tr></thead><tbody>';
    
    // Table rows (first 10)
    const rowsToShow = Math.min(10, data.length);
    for (let i = 0; i < rowsToShow; i++) {
        const row = data[i];
        previewHTML += '<tr>';
        
        columns.forEach(col => {
            const value = row[col];
            const displayValue = formatForDisplay(value);
            previewHTML += `<td>${displayValue}</td>`;
        });
        
        previewHTML += '</tr>';
    }
    
    previewHTML += '</tbody></table></div>';
    
    // Data statistics
    const statsHTML = `
        <div class="metric-box">
            <div class="metric-label">Dataset Shape:</div>
            <div class="metric-value">${totalRows} rows × ${columns.length} columns</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Training Samples:</div>
            <div class="metric-value">${data.length}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Test Samples:</div>
            <div class="metric-value">${appState.rawTestData ? appState.rawTestData.length : 0}</div>
        </div>
    `;
    
    document.getElementById('inspectionContent').innerHTML = statsHTML + previewHTML;
    createDataVisualizations();
}

function formatForDisplay(value) {
    if (value === null || value === undefined || value === '') {
        return '<span style="color: #ff5252; font-style: italic;">NULL</span>';
    }
    
    if (typeof value === 'number') {
        // Show integers without decimals, floats with 2 decimals
        return Number.isInteger(value) ? value.toString() : value.toFixed(2);
    }
    
    // Truncate long strings
    const str = String(value);
    if (str.length > 20) {
        return escapeHtml(str.substring(0, 20)) + '...';
    }
    
    return escapeHtml(str);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function createDataVisualizations() {
    const container = document.getElementById('chartsContainer');
    
    if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
        container.innerHTML = '<div class="status">No data available for visualization</div>';
        return;
    }
    
    const data = appState.rawTrainData;
    
    // Calculate survival statistics
    let totalSurvived = 0;
    let totalPassengers = 0;
    
    const survivalBySex = { male: { survived: 0, total: 0 }, female: { survived: 0, total: 0 } };
    const survivalByClass = { 1: { survived: 0, total: 0 }, 2: { survived: 0, total: 0 }, 3: { survived: 0, total: 0 } };
    
    data.forEach(row => {
        const survived = row[DATA_SCHEMA.targetColumn];
        const sex = row.Sex;
        const pclass = row.Pclass;
        
        if (survived !== undefined && survived !== null) {
            totalPassengers++;
            if (survived === 1) totalSurvived++;
        }
        
        if (sex && survivalBySex[sex.toLowerCase()]) {
            survivalBySex[sex.toLowerCase()].total++;
            if (survived === 1) survivalBySex[sex.toLowerCase()].survived++;
        }
        
        if (pclass && survivalByClass[pclass]) {
            survivalByClass[pclass].total++;
            if (survived === 1) survivalByClass[pclass].survived++;
        }
    });
    
    const survivalRate = totalPassengers > 0 ? (totalSurvived / totalPassengers * 100).toFixed(1) : 0;
    
    let html = '<h3>Data Distribution</h3>';
    
    // Overall survival
    html += `
        <div style="margin-bottom: 20px; padding: 15px; background: rgba(68, 138, 255, 0.1); border-radius: 8px;">
            <h4 style="color: #82b1ff; margin-bottom: 10px;">Overall Survival Rate</h4>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 2rem; font-weight: bold; color: #69f0ae;">${survivalRate}%</div>
                <div style="flex: 1;">
                    <div style="height: 20px; background: #37474f; border-radius: 10px; overflow: hidden;">
                        <div style="height: 100%; width: ${survivalRate}%; background: #00e676;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.9em;">
                        <span>${totalSurvived} survived</span>
                        <span>${totalPassengers} total</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Sex distribution
    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';
    
    html += '<div>';
    html += '<h4>Survival by Sex</h4>';
    Object.entries(survivalBySex).forEach(([sex, stats]) => {
        if (stats.total > 0) {
            const rate = (stats.survived / stats.total) * 100;
            html += `
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="text-transform: capitalize;">${sex}</span>
                        <span>${rate.toFixed(1)}% (${stats.survived}/${stats.total})</span>
                    </div>
                    <div style="height: 20px; background: #37474f; border-radius: 10px; overflow: hidden;">
                        <div style="height: 100%; width: ${rate}%; background: #448aff;"></div>
                    </div>
                </div>
            `;
        }
    });
    html += '</div>';
    
    // Class distribution
    html += '<div>';
    html += '<h4>Survival by Passenger Class</h4>';
    Object.entries(survivalByClass).forEach(([pclass, stats]) => {
        if (stats.total > 0) {
            const rate = (stats.survived / stats.total) * 100;
            html += `
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Class ${pclass}</span>
                        <span>${rate.toFixed(1)}% (${stats.survived}/${stats.total})</span>
                    </div>
                    <div style="height: 20px; background: #37474f; border-radius: 10px; overflow: hidden;">
                        <div style="height: 100%; width: ${rate}%; background: #ff9800;"></div>
                    </div>
                </div>
            `;
        }
    });
    html += '</div></div>';
    
    container.innerHTML = html;
}

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

function preprocessData() {
    if (!appState.rawTrainData) {
        throw new Error('No training data available');
    }
    
    calculateFeatureStatistics();
    
    // Process training data
    const processedTrain = processDataset(appState.rawTrainData, true);
    appState.processedTrainData = processedTrain;
    
    // Process test data
    if (appState.rawTestData && appState.rawTestData.length > 0) {
        const processedTest = processDataset(appState.rawTestData, false);
        appState.processedTestData = processedTest;
    }
    
    // Update UI
    document.getElementById('trainSamples').textContent = processedTrain.features.length;
    document.getElementById('valSamples').textContent = Math.floor(processedTrain.features.length * 0.2);
    
    // Show feature information
    document.getElementById('featureList').innerHTML = `
        <h4>Processed Features</h4>
        <div class="metric-box">
            <div class="metric-label">Feature Count:</div>
            <div class="metric-value">${appState.featureNames.length}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Feature Tensor Shape:</div>
            <div class="metric-value">[${processedTrain.features.length}, ${appState.featureNames.length}]</div>
        </div>
    `;
    
    updateStatus('preprocessStatus', 
        `✓ Preprocessed ${processedTrain.features.length} samples with ${appState.featureNames.length} features`, 
        'success');
    
    return processedTrain;
}

function calculateFeatureStatistics() {
    const stats = {};
    const data = appState.rawTrainData;
    
    // Age statistics
    const ageValues = data.map(row => row.Age)
        .filter(val => val !== null && !isNaN(val));
    
    if (ageValues.length > 0) {
        const mean = ageValues.reduce((a, b) => a + b, 0) / ageValues.length;
        const std = Math.sqrt(ageValues.reduce((sq, val) => sq + Math.pow(val - mean, 2), 0) / ageValues.length);
        const sorted = [...ageValues].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        stats['Age'] = { mean, std, median };
    } else {
        stats['Age'] = { mean: 0, std: 1, median: 0 };
    }
    
    // Fare statistics
    const fareValues = data.map(row => row.Fare)
        .filter(val => val !== null && !isNaN(val));
    
    if (fareValues.length > 0) {
        const mean = fareValues.reduce((a, b) => a + b, 0) / fareValues.length;
        const std = Math.sqrt(fareValues.reduce((sq, val) => sq + Math.pow(val - mean, 2), 0) / fareValues.length);
        const sorted = [...fareValues].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        stats['Fare'] = { mean, std, median };
    } else {
        stats['Fare'] = { mean: 0, std: 1, median: 0 };
    }
    
    // Categorical features mode
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        const values = data.map(row => row[feature])
            .filter(val => val !== null && val !== '');
        
        if (values.length > 0) {
            const freq = {};
            values.forEach(val => {
                const key = String(val).trim();
                freq[key] = (freq[key] || 0) + 1;
            });
            
            let mode = null;
            let maxCount = 0;
            Object.entries(freq).forEach(([value, count]) => {
                if (count > maxCount) {
                    mode = value;
                    maxCount = count;
                }
            });
            
            stats[feature] = { mode };
        } else {
            // Default modes
            if (feature === 'Embarked') stats[feature] = { mode: 'S' };
            else if (feature === 'Pclass') stats[feature] = { mode: 3 };
            else if (feature === 'Sex') stats[feature] = { mode: 'male' };
        }
    });
    
    appState.featureStats = stats;
}

function processDataset(data, isTraining) {
    const features = [];
    const labels = [];
    const passengerIds = [];
    
    data.forEach((row, index) => {
        const processed = processRow(row, isTraining);
        
        features.push(processed.featureValues);
        
        if (isTraining && row[DATA_SCHEMA.targetColumn] !== undefined) {
            labels.push(row[DATA_SCHEMA.targetColumn]);
        }
        
        passengerIds.push(row[DATA_SCHEMA.idColumn] || index + 1);
    });
    
    return { features, labels, passengerIds };
}

function processRow(row, isTraining) {
    const featureValues = [];
    
    // Store feature names on first call
    if (appState.featureNames.length === 0) {
        buildFeatureNames();
    }
    
    // Imputation values
    const ageMedian = appState.featureStats.Age?.median || 0;
    const fareMedian = appState.featureStats.Fare?.median || 0;
    const embarkedMode = appState.featureStats.Embarked?.mode || 'S';
    
    // Age (standardized)
    const age = row.Age !== null ? row.Age : ageMedian;
    const ageStd = appState.featureStats.Age?.std || 1;
    const standardizedAge = (age - ageMedian) / ageStd;
    featureValues.push(standardizedAge);
    
    // Fare (standardized)
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const fareStd = appState.featureStats.Fare?.std || 1;
    const standardizedFare = (fare - fareMedian) / fareStd;
    featureValues.push(standardizedFare);
    
    // SibSp and Parch (raw)
    featureValues.push(row.SibSp || 0);
    featureValues.push(row.Parch || 0);
    
    // Pclass one-hot
    const pclass = row.Pclass || 3;
    [1, 2, 3].forEach(cls => {
        featureValues.push(pclass === cls ? 1 : 0);
    });
    
    // Sex one-hot
    const sex = row.Sex || 'male';
    featureValues.push(sex === 'female' ? 1 : 0);
    featureValues.push(sex === 'male' ? 1 : 0);
    
    // Embarked one-hot
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    ['C', 'Q', 'S'].forEach(port => {
        featureValues.push(embarked === port ? 1 : 0);
    });
    
    // Derived features if enabled
    if (document.getElementById('addFamilyFeatures').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        featureValues.push(familySize, isAlone);
    }
    
    return { featureValues };
}

function buildFeatureNames() {
    const names = [];
    
    names.push('Age', 'Fare', 'SibSp', 'Parch');
    
    // Pclass one-hot
    [1, 2, 3].forEach(cls => {
        names.push(`Pclass_${cls}`);
    });
    
    // Sex one-hot
    names.push('Sex_female', 'Sex_male');
    
    // Embarked one-hot
    ['C', 'Q', 'S'].forEach(port => {
        names.push(`Embarked_${port}`);
    });
    
    // Derived features
    if (document.getElementById('addFamilyFeatures').checked) {
        names.push('FamilySize', 'IsAlone');
    }
    
    appState.featureNames = names;
}

// ============================================================================
// MODEL CREATION
// ============================================================================

function createModel(inputShape) {
    console.log('Creating model with input shape:', inputShape);
    
    const model = tf.sequential();
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        // Feature gate layer
        model.add(tf.layers.dense({
            units: inputShape,
            activation: 'sigmoid',
            useBias: false,
            kernelInitializer: 'ones',
            inputShape: [inputShape],
            name: 'feature_gate'
        }));
    }
    
    // Hidden layer
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelInitializer: 'heNormal',
        name: 'hidden'
    }));
    
    // Output layer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output'
    }));
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model info
    document.getElementById('modelSummary').innerHTML = `
        <h4>Model Architecture</h4>
        <div class="metric-box">
            <div class="metric-label">Input Features:</div>
            <div class="metric-value">${inputShape}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Feature Gate:</div>
            <div class="metric-value">${useFeatureGate ? 'Enabled' : 'Disabled'}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Hidden Units:</div>
            <div class="metric-value">16 (ReLU)</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Parameters:</div>
            <div class="metric-value">${model.countParams().toLocaleString()}</div>
        </div>
    `;
    
    return model;
}

// ============================================================================
// TRAINING FUNCTION
// ============================================================================

async function trainModel() {
    try {
        if (!appState.rawTrainData) {
            updateStatus('trainingStatus', 'Please load data first', 'error');
            return;
        }
        
        updateStatus('trainingStatus', 'Preprocessing data...', 'info');
        
        // Preprocess data
        const processedData = preprocessData();
        
        if (processedData.features.length === 0) {
            throw new Error('No features available after preprocessing');
        }
        
        const numFeatures = processedData.features[0].length;
        const numSamples = processedData.features.length;
        
        console.log(`Training with ${numSamples} samples, ${numFeatures} features`);
        
        // Create tensors
        const featuresTensor = tf.tensor2d(processedData.features);
        const labelsTensor = tf.tensor2d(processedData.labels, [numSamples, 1]);
        
        // Split for validation (80/20)
        const splitIndex = Math.floor(numSamples * 0.8);
        const xTrain = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const yTrain = labelsTensor.slice([0, 0], [splitIndex, -1]);
        const xVal = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const yVal = labelsTensor.slice([splitIndex, 0], [-1, -1]);
        
        // Store validation data
        appState.validationData = { xVal, yVal };
        
        // Create model
        updateStatus('trainingStatus', 'Creating model...', 'info');
        appState.model = createModel(numFeatures);
        
        // Train model
        updateStatus('trainingStatus', 'Training started (50 epochs)...', 'info');
        
        const history = await appState.model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const currentEpoch = epoch + 1;
                    updateStatus('trainingStatus', 
                        `Epoch ${currentEpoch}/50 - ` +
                        `loss: ${logs.loss.toFixed(4)}, acc: ${(logs.acc * 100).toFixed(1)}%, ` +
                        `val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${(logs.val_acc * 100).toFixed(1)}%`,
                        'success');
                    
                    appState.trainingHistory.push({
                        epoch: currentEpoch,
                        loss: logs.loss,
                        val_loss: logs.val_loss,
                        acc: logs.acc,
                        val_acc: logs.val_acc
                    });
                }
            }
        });
        
        // Clean up
        featuresTensor.dispose();
        labelsTensor.dispose();
        xTrain.dispose();
        yTrain.dispose();
        xVal.dispose();
        yVal.dispose();
        
        // Enable buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('exportModelBtn').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        
        updateStatus('trainingStatus', 
            '✓ Training completed! Click "Evaluate Model" to see metrics.', 
            'success');
        
        return history;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('trainingStatus', `✗ Training failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// EVALUATION AND METRICS
// ============================================================================

function evaluateModel() {
    if (!appState.model || !appState.validationData) {
        updateStatus('loadStatus', 'Please train the model first', 'error');
        return;
    }
    
    try {
        const { xVal, yVal } = appState.validationData;
        const predictions = appState.model.predict(xVal);
        const predValues = Array.from(predictions.dataSync());
        const trueValues = Array.from(yVal.dataSync());
        
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent) || 0.5;
        
        // Calculate confusion matrix
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        predValues.forEach((pred, idx) => {
            const predicted = pred >= threshold ? 1 : 0;
            const actual = trueValues[idx];
            
            if (actual === 1 && predicted === 1) tp++;
            else if (actual === 0 && predicted === 1) fp++;
            else if (actual === 0 && predicted === 0) tn++;
            else if (actual === 1 && predicted === 0) fn++;
        });
        
        // Update confusion matrix
        const matrixHTML = `
            <div class="confusion-header"></div>
            <div class="confusion-header">Predicted 1</div>
            <div class="confusion-header">Predicted 0</div>
            <div class="confusion-header">Actual 1</div>
            <div class="confusion-cell true-positive">${tp}</div>
            <div class="confusion-cell false-negative">${fn}</div>
            <div class="confusion-header">Actual 0</div>
            <div class="confusion-cell false-positive">${fp}</div>
            <div class="confusion-cell true-negative">${tn}</div>
        `;
        
        document.getElementById('confusionMatrix').innerHTML = matrixHTML;
        
        // Calculate metrics
        const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        const auc = calculateAUC(predValues, trueValues);
        
        // Update UI
        document.getElementById('accuracy').textContent = (accuracy * 100).toFixed(2) + '%';
        document.getElementById('precision').textContent = precision.toFixed(3);
        document.getElementById('recall').textContent = recall.toFixed(3);
        document.getElementById('f1Score').textContent = f1.toFixed(3);
        document.getElementById('aucScore').textContent = auc.toFixed(3);
        
        // Show feature importance if enabled
        const useFeatureGate = document.getElementById('useFeatureGate').checked;
        if (useFeatureGate) {
            displayFeatureImportance();
        }
        
        updateStatus('loadStatus', `✓ Evaluation completed. Accuracy: ${(accuracy * 100).toFixed(2)}%`, 'success');
        
        predictions.dispose();
        
    } catch (error) {
        updateStatus('loadStatus', `✗ Evaluation failed: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

function calculateAUC(predictions, trueLabels) {
    if (!predictions || !trueLabels || predictions.length !== trueLabels.length) {
        return 0.5;
    }
    
    // Create pairs
    const pairs = predictions.map((p, i) => ({ score: p, label: trueLabels[i] }));
    
    // Sort by score
    pairs.sort((a, b) => b.score - a.score);
    
    const totalPositives = pairs.filter(p => p.label === 1).length;
    const totalNegatives = pairs.filter(p => p.label === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) {
        return 0.5;
    }
    
    // Calculate AUC
    let auc = 0;
    let prevFPR = 0;
    let prevTPR = 0;
    let tp = 0;
    let fp = 0;
    
    const thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
    
    thresholds.forEach(threshold => {
        // Count at this threshold
        const currentPairs = pairs.filter(p => p.score >= threshold);
        tp = currentPairs.filter(p => p.label === 1).length;
        fp = currentPairs.filter(p => p.label === 0).length;
        
        const tpr = tp / totalPositives;
        const fpr = fp / totalNegatives;
        
        auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
        prevFPR = fpr;
        prevTPR = tpr;
    });
    
    return Math.max(0, Math.min(1, auc));
}

// ============================================================================
// FEATURE IMPORTANCE
// ============================================================================

function displayFeatureImportance() {
    if (!appState.model) {
        return;
    }
    
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    const container = document.getElementById('featureImportance');
    
    if (!useFeatureGate) {
        container.innerHTML = `
            <h4>Feature Importance</h4>
            <div class="status">Enable feature importance learning to see feature weights.</div>
        `;
        return;
    }
    
    try {
        const gateLayer = appState.model.layers.find(l => l.name === 'feature_gate');
        if (!gateLayer) {
            container.innerHTML = '<div class="status">Feature gate layer not found.</div>';
            return;
        }
        
        const weights = gateLayer.getWeights();
        if (weights.length === 0) {
            container.innerHTML = '<div class="status">No weights found in gate layer.</div>';
            return;
        }
        
        const kernel = weights[0];
        const kernelData = Array.from(kernel.dataSync());
        
        // Calculate importance scores
        const numFeatures = Math.min(kernel.shape[1], appState.featureNames.length);
        const importances = [];
        
        for (let i = 0; i < numFeatures; i++) {
            let sum = 0;
            for (let j = 0; j < kernel.shape[0]; j++) {
                const idx = j * kernel.shape[1] + i;
                if (idx < kernelData.length) {
                    sum += kernelData[idx];
                }
            }
            // Apply sigmoid and normalize
            const sigmoidValue = 1 / (1 + Math.exp(-sum / kernel.shape[0]));
            importances.push(sigmoidValue);
        }
        
        // Create display
        let html = '<h4>Feature Importance (Sigmoid Gate)</h4>';
        html += '<div style="max-height: 300px; overflow-y: auto;">';
        
        const featuresWithImportance = appState.featureNames.slice(0, numFeatures)
            .map((name, idx) => ({ name, importance: importances[idx] || 0 }))
            .sort((a, b) => b.importance - a.importance);
        
        featuresWithImportance.forEach((feature, idx) => {
            const width = Math.max(20, feature.importance * 100);
            const color = feature.importance > 0.7 ? '#4caf50' : 
                          feature.importance > 0.4 ? '#ff9800' : '#f44336';
            
            html += `
                <div style="margin: 8px 0; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>${idx + 1}. ${feature.name}</span>
                        <span style="color: #82b1ff; font-family: monospace;">${feature.importance.toFixed(3)}</span>
                    </div>
                    <div style="height: 6px; background: #37474f; border-radius: 3px; overflow: hidden;">
                        <div style="height: 100%; width: ${width}px; background: ${color};"></div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        container.innerHTML = `<div class="status error">Error: ${error.message}</div>`;
    }
}

// ============================================================================
// PREDICTION AND EXPORT
// ============================================================================

async function generatePredictions() {
    if (!appState.model || !appState.processedTestData) {
        updateStatus('predictionStatus', 'Please train model and load test data first', 'error');
        return;
    }
    
    try {
        const testFeatures = tf.tensor2d(appState.processedTestData.features);
        const predictions = appState.model.predict(testFeatures);
        const probabilities = Array.from(predictions.dataSync());
        
        appState.testPredictions = probabilities;
        appState.testPassengerIds = appState.processedTestData.passengerIds;
        
        // Show sample predictions
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent) || 0.5;
        let sampleHTML = '<h4>Sample Predictions (First 5)</h4>';
        sampleHTML += '<table><tr><th>PassengerId</th><th>Probability</th><th>Predicted</th></tr>';
        
        const sampleSize = Math.min(5, probabilities.length);
        for (let i = 0; i < sampleSize; i++) {
            const prob = probabilities[i];
            const predClass = prob >= threshold ? 1 : 0;
            sampleHTML += `
                <tr>
                    <td>${appState.testPassengerIds[i]}</td>
                    <td>${prob.toFixed(4)}</td>
                    <td><strong>${predClass}</strong> ${predClass === 1 ? '✓ Survived' : '✗ Not survived'}</td>
                </tr>
            `;
        }
        
        sampleHTML += '</table>';
        sampleHTML += `<p>Total predictions generated: <strong>${probabilities.length}</strong></p>`;
        
        updateStatus('predictionStatus', sampleHTML, 'success');
        
        // Enable export buttons
        document.getElementById('exportSubmissionBtn').disabled = false;
        document.getElementById('exportProbabilitiesBtn').disabled = false;
        
        testFeatures.dispose();
        predictions.dispose();
        
    } catch (error) {
        updateStatus('predictionStatus', `✗ Prediction failed: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

function exportSubmissionCSV() {
    if (!appState.testPredictions || !appState.testPassengerIds) {
        alert('Please generate predictions first');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('thresholdValue').textContent) || 0.5;
    let csv = 'PassengerId,Survived\n';
    
    appState.testPredictions.forEach((prob, idx) => {
        const survived = prob >= threshold ? 1 : 0;
        csv += `${appState.testPassengerIds[idx]},${survived}\n`;
    });
    
    downloadCSV(csv, 'submission.csv');
    updateStatus('predictionStatus', '✓ submission.csv downloaded', 'success');
}

function exportProbabilitiesCSV() {
    if (!appState.testPredictions || !appState.testPassengerIds) {
        alert('Please generate predictions first');
        return;
    }
    
    let csv = 'PassengerId,Probability\n';
    
    appState.testPredictions.forEach((prob, idx) => {
        csv += `${appState.testPassengerIds[idx]},${prob.toFixed(6)}\n`;
    });
    
    downloadCSV(csv, 'probabilities.csv');
    updateStatus('predictionStatus', '✓ probabilities.csv downloaded', 'success');
}

async function exportModel() {
    if (!appState.model) {
        alert('Please train a model first');
        return;
    }
    
    try {
        await appState.model.save('downloads://titanic-model');
        updateStatus('trainingStatus', '✓ Model downloaded successfully!', 'success');
    } catch (error) {
        updateStatus('trainingStatus', `✗ Failed to export model: ${error.message}`, 'error');
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    document.getElementById('loadBtn').addEventListener('click', loadData);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', generatePredictions);
    
    document.getElementById('thresholdSlider').addEventListener('input', (e) => {
        const threshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
        if (appState.validationData) {
            evaluateModel();
        }
    });
    
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmissionCSV);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilitiesCSV);
    document.getElementById('exportModelBtn').addEventListener('click', exportModel);
    
    // File inputs
    document.getElementById('trainFile').addEventListener('change', () => {
        document.getElementById('useDemoData').checked = false;
    });
    
    document.getElementById('testFile').addEventListener('change', () => {
        document.getElementById('useDemoData').checked = false;
    });
    
    // Feature toggles
    document.getElementById('addFamilyFeatures').addEventListener('change', () => {
        // Reset feature names if preprocessing was already done
        appState.featureNames = [];
    });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    console.log('Titanic Classifier initialized');
});
