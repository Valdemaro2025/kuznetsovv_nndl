// app.js - Fixed Titanic Classifier

// ============================================================================
// GLOBAL STATE AND CONFIGURATION
// ============================================================================

// Global application state
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

// Schema configuration - MODIFY THIS FOR OTHER DATASETS
const DATA_SCHEMA = {
    // Target variable (binary classification)
    targetColumn: 'Survived',
    
    // Identifier column (excluded from features)
    idColumn: 'PassengerId',
    
    // Numerical features (will be standardized)
    numericalFeatures: ['Age', 'Fare'],
    
    // Features that will be used as-is (no standardization)
    rawNumericalFeatures: ['SibSp', 'Parch'],
    
    // Categorical features (will be one-hot encoded)
    categoricalFeatures: ['Pclass', 'Sex', 'Embarked'],
    
    // Columns to exclude completely
    excludeColumns: ['Name', 'Ticket', 'Cabin'],
    
    // Derived features configuration
    derivedFeatures: {
        'FamilySize': (row) => (row.SibSp || 0) + (row.Parch || 0) + 1,
        'IsAlone': (row) => ((row.SibSp || 0) + (row.Parch || 0) === 0) ? 1 : 0
    }
};

// ============================================================================
// CORE UTILITY FUNCTIONS - FIXED
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
// CSV FILE PARSING - FIXED FOR COMMA ESCAPE PROBLEM
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
                
                // Handle quoted fields with commas inside
                const lines = text.split(/\r\n|\n|\r/).filter(line => line.trim() !== '');
                
                if (lines.length < 2) {
                    reject(new Error('CSV file must have at least a header and one data row'));
                    return;
                }
                
                // Parse headers
                const headers = parseCSVLine(lines[0]);
                
                // Parse data rows
                const rows = [];
                for (let i = 1; i < lines.length; i++) {
                    const values = parseCSVLine(lines[i]);
                    const row = {};
                    
                    headers.forEach((header, index) => {
                        let value = values[index] || '';
                        value = value.trim();
                        
                        // Handle missing values
                        if (value === '' || value === 'NULL' || value === 'null') {
                            value = null;
                        } else {
                            // Try to convert to number if possible
                            const numValue = parseFloat(value);
                            if (!isNaN(numValue)) {
                                value = numValue;
                            }
                        }
                        
                        row[header] = value;
                    });
                    
                    rows.push(row);
                }
                
                console.log(`Parsed ${rows.length} rows with ${headers.length} columns`);
                console.log('First row sample:', rows[0]);
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

// FIX: Proper CSV line parsing with quoted fields
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            // Handle escaped quotes
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
    return result.map(val => val.replace(/^"|"$/g, '').trim());
}

// ============================================================================
// DATA LOADING FUNCTIONS
// ============================================================================

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
        const useDemoData = document.getElementById('useDemoData').checked;
        
        updateStatus('loadStatus', 'Loading data...', 'info');
        
        if (useDemoData) {
            // Load demo data
            await loadDemoData();
            updateStatus('loadStatus', 
                `Loaded demo data: ${appState.rawTrainData.length} training samples`, 
                'success');
        } else if (trainFile) {
            // Parse uploaded files
            appState.rawTrainData = await parseCSVFile(trainFile);
            
            if (testFile) {
                appState.rawTestData = await parseCSVFile(testFile);
            } else {
                // Create test split from training data
                const testSize = Math.floor(appState.rawTrainData.length * 0.2);
                appState.rawTestData = appState.rawTrainData.slice(-testSize);
                appState.rawTrainData = appState.rawTrainData.slice(0, -testSize);
            }
            
            updateStatus('loadStatus', 
                `Loaded ${appState.rawTrainData.length} training and ${appState.rawTestData ? appState.rawTestData.length : 0} test samples`, 
                'success');
        } else {
            updateStatus('loadStatus', 'Please upload train.csv or check "Use demo data"', 'error');
            return false;
        }
        
        // Enable training button
        document.getElementById('trainBtn').disabled = false;
        
        // Display data inspection
        inspectData();
        
        return true;
        
    } catch (error) {
        updateStatus('loadStatus', `Failed to load data: ${error.message}`, 'error');
        console.error('Load error:', error);
        return false;
    }
}

async function loadDemoData() {
    // Create realistic demo Titanic data (fixed from image)
    appState.rawTrainData = [
        {PassengerId: 1, Survived: 0, Pclass: 3, Name: 'Braund, Mr. Owen Harris', Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Ticket: 'A/5 21171', Fare: 7.25, Cabin: null, Embarked: 'S'},
        {PassengerId: 2, Survived: 1, Pclass: 1, Name: 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Ticket: 'PC 17599', Fare: 71.28, Cabin: 'C85', Embarked: 'C'},
        {PassengerId: 3, Survived: 1, Pclass: 3, Name: 'Heikkinen, Miss. Laina', Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Ticket: 'STON/O2. 3101282', Fare: 7.92, Cabin: null, Embarked: 'S'},
        {PassengerId: 4, Survived: 1, Pclass: 1, Name: 'Futrelle, Mrs. Jacques Heath (Lily May Peel)', Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Ticket: '113803', Fare: 53.10, Cabin: 'C123', Embarked: 'S'},
        {PassengerId: 5, Survived: 0, Pclass: 3, Name: 'Allen, Mr. William Henry', Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Ticket: '373450', Fare: 8.05, Cabin: null, Embarked: 'S'},
        {PassengerId: 6, Survived: 0, Pclass: 3, Name: 'Moran, Mr. James', Sex: 'male', Age: null, SibSp: 0, Parch: 0, Ticket: '330877', Fare: 8.46, Cabin: null, Embarked: 'Q'},
        {PassengerId: 7, Survived: 0, Pclass: 1, Name: 'McCarthy, Mr. Timothy J', Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Ticket: '17463', Fare: 51.86, Cabin: 'E46', Embarked: 'S'},
        {PassengerId: 8, Survived: 0, Pclass: 3, Name: 'Palsson, Master. Gosta Leonard', Sex: 'male', Age: 2, SibSp: 3, Parch: 1, Ticket: '349909', Fare: 21.07, Cabin: null, Embarked: 'S'},
        {PassengerId: 9, Survived: 1, Pclass: 3, Name: 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)', Sex: 'female', Age: 27, SibSp: 0, Parch: 2, Ticket: '347742', Fare: 11.13, Cabin: null, Embarked: 'S'},
        {PassengerId: 10, Survived: 1, Pclass: 2, Name: 'Nasser, Mrs. Nicholas (Adele Achem)', Sex: 'female', Age: 14, SibSp: 1, Parch: 0, Ticket: '237736', Fare: 30.07, Cabin: null, Embarked: 'C'}
    ];
    
    appState.rawTestData = [
        {PassengerId: 11, Pclass: 3, Name: 'Sandstrom, Miss. Marguerite Rut', Sex: 'female', Age: 4, SibSp: 1, Parch: 1, Ticket: 'PP 9549', Fare: 16.70, Cabin: 'G6', Embarked: 'S'},
        {PassengerId: 12, Pclass: 1, Name: 'Bonnell, Miss. Elizabeth', Sex: 'female', Age: 58, SibSp: 0, Parch: 0, Ticket: '113783', Fare: 26.55, Cabin: 'C103', Embarked: 'S'},
        {PassengerId: 13, Pclass: 3, Name: 'Saundercock, Mr. William Henry', Sex: 'male', Age: 20, SibSp: 0, Parch: 0, Ticket: 'A/5. 2151', Fare: 8.05, Cabin: null, Embarked: 'S'},
        {PassengerId: 14, Pclass: 3, Name: 'Andersson, Mr. Anders Johan', Sex: 'male', Age: 39, SibSp: 1, Parch: 5, Ticket: '347082', Fare: 31.28, Cabin: null, Embarked: 'S'},
        {PassengerId: 15, Pclass: 3, Name: 'Vestrom, Miss. Hulda Amanda Adolfina', Sex: 'female', Age: 14, SibSp: 0, Parch: 0, Ticket: '350406', Fare: 7.85, Cabin: null, Embarked: 'S'}
    ];
    
    return true;
}

// ============================================================================
// DATA INSPECTION AND VISUALIZATION - FIXED
// ============================================================================

function inspectData() {
    if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
        updateStatus('inspectionContent', 'No data loaded', 'error');
        return;
    }
    
    const data = appState.rawTrainData;
    const columns = Object.keys(data[0]);
    const totalRows = data.length;
    
    // Create data preview (as shown in image)
    let previewHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    
    // Create table for key columns (as in image)
    previewHTML += `
        <div class="data-preview">
        <table>
            <thead>
                <tr>
                    <th>PassengerID</th>
                    <th>Survived</th>
                    <th>Pclass</th>
                    <th>Name</th>
                    <th>Sex</th>
                    <th>Age</th>
                    <th>SibSp</th>
                    <th>Parch</th>
                    <th>Ticket</th>
                    <th>Fare</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    data.slice(0, 10).forEach(row => {
        previewHTML += `
            <tr>
                <td>${row.PassengerId || ''}</td>
                <td>${row.Survived !== null ? row.Survived : 'NULL'}</td>
                <td>${row.Pclass || ''}</td>
                <td>${(row.Name || '').substring(0, 15)}${(row.Name || '').length > 15 ? '...' : ''}</td>
                <td>${row.Sex || ''}</td>
                <td>${row.Age !== null ? row.Age : 'NULL'}</td>
                <td>${row.SibSp || 0}</td>
                <td>${row.Parch || 0}</td>
                <td>${(row.Ticket || '').substring(0, 10)}${(row.Ticket || '').length > 10 ? '...' : ''}</td>
                <td>${row.Fare !== null ? row.Fare.toFixed(2) : 'NULL'}</td>
            </tr>
        `;
    });
    
    previewHTML += '</tbody></table></div>';
    
    // Update inspection content
    document.getElementById('inspectionContent').innerHTML = `
        <div class="metric-box">
            <div class="metric-label">Dataset Shape:</div>
            <div class="metric-value">${totalRows} rows × ${columns.length} columns</div>
        </div>
        ${previewHTML}
    `;
    
    // Create visualizations
    createDataVisualizations();
}

function createDataVisualizations() {
    const container = document.getElementById('chartsContainer');
    
    if (!appState.rawTrainData) {
        container.innerHTML = '<p>No data available for visualization</p>';
        return;
    }
    
    // Calculate survival statistics
    const survivalBySex = { male: { survived: 0, total: 0 }, female: { survived: 0, total: 0 } };
    const survivalByClass = { 1: { survived: 0, total: 0 }, 2: { survived: 0, total: 0 }, 3: { survived: 0, total: 0 } };
    
    appState.rawTrainData.forEach(row => {
        const sex = row.Sex;
        const pclass = row.Pclass;
        const survived = row.Survived;
        
        if (sex && survivalBySex[sex.toLowerCase()]) {
            survivalBySex[sex.toLowerCase()].total++;
            if (survived === 1) survivalBySex[sex.toLowerCase()].survived++;
        }
        
        if (pclass && survivalByClass[pclass]) {
            survivalByClass[pclass].total++;
            if (survived === 1) survivalByClass[pclass].survived++;
        }
    });
    
    // Create HTML visualizations
    let html = '<h3>Data Distribution</h3><div style="display: flex; flex-wrap: wrap; gap: 20px;">';
    
    // Sex distribution
    html += '<div style="flex: 1; min-width: 300px;">';
    html += '<h4>Survival by Sex</h4>';
    Object.entries(survivalBySex).forEach(([sex, stats]) => {
        if (stats.total > 0) {
            const rate = (stats.survived / stats.total) * 100;
            html += `
                <div style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${sex}</span>
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
    html += '<div style="flex: 1; min-width: 300px;">';
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
                        <div style="height: 100%; width: ${rate}%; background: #00e676;"></div>
                    </div>
                </div>
            `;
        }
    });
    html += '</div></div>';
    
    container.innerHTML = html;
}

// ============================================================================
// DATA PREPROCESSING - FIXED TO MATCH TEACHER'S CODE
// ============================================================================

function preprocessData() {
    if (!appState.rawTrainData) {
        throw new Error('No training data available');
    }
    
    // Calculate statistics from training data
    calculateFeatureStatistics();
    
    // Process training data
    const processedTrain = processDataset(appState.rawTrainData, true);
    appState.processedTrainData = processedTrain;
    
    // Process test data if available
    if (appState.rawTestData && appState.rawTestData.length > 0) {
        const processedTest = processDataset(appState.rawTestData, false);
        appState.processedTestData = processedTest;
    }
    
    // Show feature information
    document.getElementById('featureList').innerHTML = `
        <h3>Processed Features (${appState.featureNames.length})</h3>
        <p><small>${appState.featureNames.join(', ')}</small></p>
    `;
    
    updateStatus('preprocessStatus', 
        `Preprocessed ${processedTrain.features.length} samples with ${appState.featureNames.length} features`, 
        'success');
    
    return processedTrain;
}

function calculateFeatureStatistics() {
    const stats = {};
    const data = appState.rawTrainData;
    
    // Age statistics (for imputation)
    const ageValues = data.map(row => row.Age)
        .filter(val => val !== null && !isNaN(val) && val !== '');
    
    if (ageValues.length > 0) {
        const mean = ageValues.reduce((a, b) => a + b, 0) / ageValues.length;
        const std = Math.sqrt(ageValues.reduce((sq, val) => sq + Math.pow(val - mean, 2), 0) / ageValues.length);
        const sorted = [...ageValues].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        stats['Age'] = { mean, std, median, type: 'numerical' };
    } else {
        stats['Age'] = { mean: 0, std: 1, median: 0, type: 'numerical' };
    }
    
    // Fare statistics
    const fareValues = data.map(row => row.Fare)
        .filter(val => val !== null && !isNaN(val) && val !== '');
    
    if (fareValues.length > 0) {
        const mean = fareValues.reduce((a, b) => a + b, 0) / fareValues.length;
        const std = Math.sqrt(fareValues.reduce((sq, val) => sq + Math.pow(val - mean, 2), 0) / fareValues.length);
        const sorted = [...fareValues].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        stats['Fare'] = { mean, std, median, type: 'numerical' };
    } else {
        stats['Fare'] = { mean: 0, std: 1, median: 0, type: 'numerical' };
    }
    
    // Categorical features mode calculation
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        const values = data.map(row => row[feature])
            .filter(val => val !== null && val !== '');
        
        if (values.length > 0) {
            // Count frequencies
            const freq = {};
            values.forEach(val => {
                const key = String(val).trim();
                freq[key] = (freq[key] || 0) + 1;
            });
            
            // Find mode (most common value)
            let mode = null;
            let maxCount = 0;
            Object.entries(freq).forEach(([value, count]) => {
                if (count > maxCount) {
                    mode = value;
                    maxCount = count;
                }
            });
            
            stats[feature] = { 
                mode, 
                frequencies: freq,
                categories: Object.keys(freq),
                type: 'categorical' 
            };
        } else {
            // Default modes from Titanic dataset
            if (feature === 'Embarked') stats[feature] = { mode: 'S', type: 'categorical' };
            else if (feature === 'Pclass') stats[feature] = { mode: 3, type: 'categorical' };
            else if (feature === 'Sex') stats[feature] = { mode: 'male', type: 'categorical' };
        }
    });
    
    appState.featureStats = stats;
    console.log('Feature statistics calculated:', stats);
}

function processDataset(data, isTraining) {
    const features = [];
    const labels = [];
    const passengerIds = [];
    
    // Get feature names from first processed row
    let featureNames = [];
    
    data.forEach((row, index) => {
        const processed = processRow(row, isTraining);
        
        if (index === 0) {
            featureNames = processed.featureNames;
        }
        
        features.push(processed.featureValues);
        
        if (isTraining && row[DATA_SCHEMA.targetColumn] !== undefined) {
            labels.push(row[DATA_SCHEMA.targetColumn]);
        }
        
        passengerIds.push(row[DATA_SCHEMA.idColumn] || index + 1);
    });
    
    // Store feature names globally
    if (featureNames.length > 0 && appState.featureNames.length === 0) {
        appState.featureNames = featureNames;
    }
    
    return {
        features,
        labels,
        passengerIds,
        featureNames
    };
}

function processRow(row, isTraining) {
    const featureValues = [];
    const featureNames = [];
    
    // IMITATION: Handle missing values like teacher's code
    const ageMedian = appState.featureStats.Age?.median || 0;
    const fareMedian = appState.featureStats.Fare?.median || 0;
    const embarkedMode = appState.featureStats.Embarked?.mode || 'S';
    
    // Age with imputation and standardization
    const age = row.Age !== null ? row.Age : ageMedian;
    const ageStd = appState.featureStats.Age?.std || 1;
    const standardizedAge = (age - ageMedian) / ageStd;
    featureValues.push(standardizedAge);
    featureNames.push('Age');
    
    // Fare with imputation and standardization
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const fareStd = appState.featureStats.Fare?.std || 1;
    const standardizedFare = (fare - fareMedian) / fareStd;
    featureValues.push(standardizedFare);
    featureNames.push('Fare');
    
    // SibSp and Parch (no standardization, used as-is)
    featureValues.push(row.SibSp || 0);
    featureNames.push('SibSp');
    featureValues.push(row.Parch || 0);
    featureNames.push('Parch');
    
    // One-hot encoding for Pclass (1, 2, 3)
    const pclass = row.Pclass || 3;
    [1, 2, 3].forEach(cls => {
        featureValues.push(pclass === cls ? 1 : 0);
        featureNames.push(`Pclass_${cls}`);
    });
    
    // One-hot encoding for Sex
    const sex = row.Sex || 'male';
    featureValues.push(sex === 'female' ? 1 : 0);
    featureNames.push('Sex_female');
    featureValues.push(sex === 'male' ? 1 : 0);
    featureNames.push('Sex_male');
    
    // One-hot encoding for Embarked
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    ['C', 'Q', 'S'].forEach(port => {
        featureValues.push(embarked === port ? 1 : 0);
        featureNames.push(`Embarked_${port}`);
    });
    
    // Add derived features if enabled
    if (document.getElementById('addFamilyFeatures').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        featureValues.push(familySize, isAlone);
        featureNames.push('FamilySize', 'IsAlone');
    }
    
    return { featureValues, featureNames };
}

// ============================================================================
// MODEL CREATION - WITH SIGMOID GATE
// ============================================================================

function createModel(inputShape) {
    console.log('Creating model with', inputShape, 'input features');
    
    const model = tf.sequential();
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        // Sigmoid gate layer for feature importance
        model.add(tf.layers.dense({
            units: inputShape, // Same as input for element-wise importance
            activation: 'sigmoid', // Values between 0-1
            useBias: false, // No bias for gate
            kernelInitializer: 'ones', // Start with all gates open
            inputShape: [inputShape],
            name: 'feature_gate'
        }));
        
        // Hidden layer
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'hidden'
        }));
    } else {
        // Without feature gate
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape],
            kernelInitializer: 'heNormal',
            name: 'hidden'
        }));
    }
    
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
    const modelSummaryDiv = document.getElementById('modelSummary');
    modelSummaryDiv.innerHTML = `
        <h3>Model Architecture</h3>
        <div style="background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; margin: 10px 0;">
            <div><strong>Input Features:</strong> ${inputShape}</div>
            <div><strong>Feature Gate:</strong> ${useFeatureGate ? 'Enabled (sigmoid)' : 'Disabled'}</div>
            <div><strong>Hidden Layer:</strong> 16 units (ReLU)</div>
            <div><strong>Output Layer:</strong> 1 unit (sigmoid)</div>
            <div><strong>Total Parameters:</strong> ${model.countParams().toLocaleString()}</div>
        </div>
    `;
    
    console.log('Model created successfully');
    return model;
}

// ============================================================================
// TRAINING FUNCTION - FIXED
// ============================================================================

async function trainModel() {
    try {
        // Check if data is loaded
        if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
            updateStatus('trainingStatus', 'Please load data first', 'error');
            return;
        }
        
        updateStatus('trainingStatus', 'Preprocessing data...', 'info');
        
        // Preprocess data
        const processedData = preprocessData();
        
        if (!processedData.features || processedData.features.length === 0) {
            throw new Error('No features available after preprocessing');
        }
        
        const numFeatures = processedData.features[0].length;
        const numSamples = processedData.features.length;
        
        console.log(`Training with ${numSamples} samples, ${numFeatures} features`);
        
        // Create tensors
        const featuresTensor = tf.tensor2d(processedData.features);
        const labelsTensor = tf.tensor2d(processedData.labels, [numSamples, 1]);
        
        // Create validation split (80/20)
        const splitIndex = Math.floor(numSamples * 0.8);
        
        const xTrain = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const yTrain = labelsTensor.slice([0, 0], [splitIndex, -1]);
        const xVal = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const yVal = labelsTensor.slice([splitIndex, 0], [-1, -1]);
        
        // Store validation data for later evaluation
        appState.validationData = { 
            xVal: xVal, 
            yVal: yVal 
        };
        
        // Create model
        updateStatus('trainingStatus', 'Creating model...', 'info');
        appState.model = createModel(numFeatures);
        
        // Train model for 50 epochs (as in image)
        updateStatus('trainingStatus', 'Training started (50 epochs)...', 'info');
        
        const history = await appState.model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const currentEpoch = epoch + 1;
                    const progress = Math.round((currentEpoch / 50) * 100);
                    
                    updateStatus('trainingStatus', 
                        `Epoch ${currentEpoch}/50 (${progress}%) - ` +
                        `loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, ` +
                        `val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`,
                        'success');
                    
                    // Store history
                    appState.trainingHistory.push({
                        epoch: currentEpoch,
                        loss: logs.loss,
                        val_loss: logs.val_loss,
                        acc: logs.acc,
                        val_acc: logs.val_acc
                    });
                },
                onTrainEnd: () => {
                    console.log('Training completed');
                    // Show final accuracy from image (64.25%)
                    const finalValAcc = appState.trainingHistory[appState.trainingHistory.length - 1]?.val_acc || 0;
                    document.getElementById('modelSummary').innerHTML += `
                        <div class="metric-box" style="margin-top: 15px;">
                            <div class="metric-label">Final Validation Accuracy:</div>
                            <div class="metric-value">${(finalValAcc * 100).toFixed(2)}%</div>
                        </div>
                    `;
                }
            }
        });
        
        console.log('Training completed successfully');
        
        // Enable evaluation and export buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('exportModelBtn').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        
        updateStatus('trainingStatus', 
            'Training completed successfully! Click "Evaluate Model" to see metrics.', 
            'success');
        
        return history;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('trainingStatus', `Training failed: ${error.message}`, 'error');
    }
}

// ============================================================================
// EVALUATION AND METRICS - FIXED TO SHOW EVALUATION TABLE
// ============================================================================

function evaluateModel() {
    if (!appState.model || !appState.validationData) {
        updateStatus('loadStatus', 'Please train the model first', 'error');
        return;
    }
    
    try {
        const { xVal, yVal } = appState.validationData;
        
        // Get predictions
        const predictions = appState.model.predict(xVal);
        const predValues = Array.from(predictions.dataSync());
        const trueValues = Array.from(yVal.dataSync());
        
        // Update threshold display
        const threshold = 0.5;
        document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
        
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
        
        // FIXED: Update confusion matrix display
        const matrixHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px; margin: 20px 0;">
                <div class="confusion-header"></div>
                <div class="confusion-header">Predicted Positive</div>
                <div class="confusion-header">Predicted Negative</div>
                <div class="confusion-header">Actual Positive</div>
                <div class="confusion-cell true-positive">${tp}</div>
                <div class="confusion-cell false-negative">${fn}</div>
                <div class="confusion-header">Actual Negative</div>
                <div class="confusion-cell false-positive">${fp}</div>
                <div class="confusion-cell true-negative">${tn}</div>
            </div>
        `;
        
        document.getElementById('confusionMatrix').innerHTML = matrixHTML;
        
        // Calculate metrics
        const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;
        
        // Update metrics display
        document.getElementById('accuracy').textContent = (accuracy * 100).toFixed(2) + '%';
        document.getElementById('precision').textContent = precision.toFixed(3);
        document.getElementById('recall').textContent = recall.toFixed(3);
        document.getElementById('f1Score').textContent = f1.toFixed(3);
        
        // Calculate AUC
        const auc = calculateAUC(predValues, trueValues);
        document.getElementById('aucScore').textContent = auc.toFixed(3);
        
        // Display feature importance if gate is enabled
        const useFeatureGate = document.getElementById('useFeatureGate').checked;
        if (useFeatureGate) {
            displayFeatureImportance();
        }
        
        updateStatus('loadStatus', `Evaluation completed. Accuracy: ${(accuracy * 100).toFixed(2)}%`, 'success');
        
        predictions.dispose();
        
    } catch (error) {
        updateStatus('loadStatus', `Evaluation failed: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

// ============================================================================
// FEATURE IMPORTANCE - SIGMOID GATE ANALYSIS
// ============================================================================

function displayFeatureImportance() {
    if (!appState.model) return;
    
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    const container = document.getElementById('featureImportance');
    
    if (!useFeatureGate) {
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <div class="status">
                <i class="fas fa-info-circle"></i> Feature gate is disabled.
            </div>
        `;
        return;
    }
    
    try {
        // Get feature gate layer weights
        const gateLayer = appState.model.layers.find(l => l.name === 'feature_gate');
        if (!gateLayer) return;
        
        const weights = gateLayer.getWeights();
        if (weights.length === 0) return;
        
        // Get the kernel weights
        const kernel = weights[0];
        const kernelData = Array.from(kernel.dataSync());
        
        // For a diagonal-like matrix, take average of each column
        const numFeatures = Math.min(kernel.shape[1], appState.featureNames.length);
        let importances = [];
        
        for (let i = 0; i < numFeatures; i++) {
            let sum = 0;
            for (let j = 0; j < kernel.shape[0]; j++) {
                const idx = j * kernel.shape[1] + i;
                if (idx < kernelData.length) {
                    sum += kernelData[idx];
                }
            }
            // Apply sigmoid (gate activation) and normalize
            const sigmoidValue = 1 / (1 + Math.exp(-sum / kernel.shape[0]));
            importances.push(sigmoidValue);
        }
        
        // Create feature importance display
        let html = '<h3>Feature Importance (Sigmoid Gate Weights)</h3><table>';
        html += '<tr><th>Feature</th><th>Importance</th><th>Bar</th></tr>';
        
        const featuresWithImportance = appState.featureNames.slice(0, numFeatures).map((name, idx) => ({
            name,
            importance: importances[idx] || 0
        }));
        
        // Sort by importance
        featuresWithImportance.sort((a, b) => b.importance - a.importance);
        
        featuresWithImportance.forEach(feature => {
            const width = Math.max(5, feature.importance * 100);
            const color = feature.importance > 0.7 ? '#4caf50' : 
                          feature.importance > 0.4 ? '#ff9800' : '#f44336';
            
            html += `
                <tr>
                    <td>${feature.name}</td>
                    <td>${feature.importance.toFixed(3)}</td>
                    <td>
                        <div style="width: ${width}px; height: 10px; background: ${color}; border-radius: 2px;"></div>
                    </td>
                </tr>
            `;
        });
        
        html += '</table>';
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        container.innerHTML = `<h3>Feature Importance</h3><p>Error: ${error.message}</p>`;
    }
}

// ============================================================================
// PREDICTION AND EXPORT - FIXED .toFixed ERROR
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
        
        // Show sample predictions (FIXED: no .toFixed on strings)
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent) || 0.5;
        let sampleHTML = '<h3>Sample Predictions (First 5)</h3><table>';
        sampleHTML += '<tr><th>PassengerId</th><th>Probability</th><th>Predicted (≥0.5)</th></tr>';
        
        const sampleSize = Math.min(5, probabilities.length);
        for (let i = 0; i < sampleSize; i++) {
            const prob = probabilities[i];
            const predClass = prob >= threshold ? 1 : 0;
            sampleHTML += `
                <tr>
                    <td>${appState.testPassengerIds[i]}</td>
                    <td>${typeof prob === 'number' ? prob.toFixed(4) : prob}</td>
                    <td>${predClass}</td>
                </tr>
            `;
        }
        
        sampleHTML += '</table>';
        sampleHTML += `<p>Total predictions: ${probabilities.length}</p>`;
        
        updateStatus('predictionStatus', sampleHTML, 'success');
        
        // Enable export buttons
        document.getElementById('exportSubmissionBtn').disabled = false;
        document.getElementById('exportProbabilitiesBtn').disabled = false;
        
        testFeatures.dispose();
        predictions.dispose();
        
    } catch (error) {
        updateStatus('predictionStatus', `Prediction failed: ${error.message}`, 'error');
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
}

// ============================================================================
// AUC CALCULATION
// ============================================================================

function calculateAUC(predictions, trueLabels) {
    if (!predictions || !trueLabels || predictions.length !== trueLabels.length) {
        return 0.5;
    }
    
    // Create pairs
    const pairs = predictions.map((p, i) => ({ score: p, label: trueLabels[i] }));
    
    // Sort by score (descending)
    pairs.sort((a, b) => b.score - a.score);
    
    const totalPositives = pairs.filter(p => p.label === 1).length;
    const totalNegatives = pairs.filter(p => p.label === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) {
        return 0.5;
    }
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    let prevFPR = 0;
    let prevTPR = 0;
    let tp = 0;
    let fp = 0;
    
    // Get unique thresholds from sorted scores
    const thresholds = [...new Set(pairs.map(p => p.score))].sort((a, b) => b - a);
    
    for (const threshold of thresholds) {
        // Count TP and FP at this threshold
        while (pairs.length > 0 && pairs[0].score >= threshold) {
            const pair = pairs.shift();
            if (pair.label === 1) tp++;
            else fp++;
        }
        
        const tpr = tp / totalPositives;
        const fpr = fp / totalNegatives;
        
        // Add trapezoid area
        auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
        
        prevFPR = fpr;
        prevTPR = tpr;
    }
    
    return Math.max(0, Math.min(1, auc));
}

// ============================================================================
// EVENT LISTENERS AND INITIALIZATION
// ============================================================================

function setupEventListeners() {
    // Load data button
    document.getElementById('loadBtn').addEventListener('click', loadData);
    
    // Train model button
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    
    // Evaluate button
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', generatePredictions);
    
    // Threshold slider
    document.getElementById('thresholdSlider').addEventListener('input', (e) => {
        const threshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
        if (appState.validationData) {
            evaluateModel(); // Re-evaluate with new threshold
        }
    });
    
    // Export buttons
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmissionCSV);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilitiesCSV);
    document.getElementById('exportModelBtn').addEventListener('click', async () => {
        if (appState.model) {
            await appState.model.save('downloads://titanic-model');
            updateStatus('trainingStatus', 'Model downloaded successfully!', 'success');
        }
    });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateStatus('loadStatus', 'Ready to load Titanic dataset. Check "Use demo data" for instant demo.', 'info');
});
