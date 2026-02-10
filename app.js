// app.js - Complete Titanic Classifier with Fixed Architecture

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
    numericalFeatures: ['Age', 'Fare', 'SibSp', 'Parch'],
    
    // Categorical features (will be one-hot encoded)
    categoricalFeatures: ['Sex', 'Pclass', 'Embarked'],
    
    // Columns to exclude completely
    excludeColumns: ['Name', 'Ticket', 'Cabin'],
    
    // Derived features configuration
    derivedFeatures: {
        'FamilySize': (row) => (row.SibSp || 0) + (row.Parch || 0) + 1,
        'IsAlone': (row) => ((row.SibSp || 0) + (row.Parch || 0) === 0) ? 1 : 0
    }
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
// CSV FILE PARSING
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
                
                // Handle different line endings
                const lines = text.split(/\r\n|\n|\r/).filter(line => line.trim() !== '');
                
                if (lines.length < 2) {
                    reject(new Error('CSV file must have at least a header and one data row'));
                    return;
                }
                
                // Parse headers
                const headers = lines[0].split(',').map(h => h.trim());
                
                // Parse data rows
                const rows = [];
                for (let i = 1; i < lines.length; i++) {
                    const values = lines[i].split(',');
                    const row = {};
                    
                    headers.forEach((header, index) => {
                        let value = values[index] || '';
                        value = value.trim().replace(/^"|"$/g, '');
                        
                        // Convert numeric values
                        if (!isNaN(value) && value !== '') {
                            value = parseFloat(value);
                        } else if (value === '') {
                            value = null;
                        }
                        
                        row[header] = value;
                    });
                    
                    rows.push(row);
                }
                
                console.log(`Parsed ${rows.length} rows with ${headers.length} columns`);
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
                `Loaded demo data: ${appState.rawTrainData.length} training, ${appState.rawTestData ? appState.rawTestData.length : 0} test samples`, 
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
                `Loaded ${appState.rawTrainData.length} training and ${appState.rawTestData.length} test samples`, 
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
    // Create realistic demo Titanic data
    appState.rawTrainData = [
        {PassengerId: 1, Survived: 0, Pclass: 3, Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Fare: 7.25, Embarked: 'S'},
        {PassengerId: 2, Survived: 1, Pclass: 1, Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Fare: 71.28, Embarked: 'C'},
        {PassengerId: 3, Survived: 1, Pclass: 3, Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Fare: 7.92, Embarked: 'S'},
        {PassengerId: 4, Survived: 1, Pclass: 1, Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Fare: 53.10, Embarked: 'S'},
        {PassengerId: 5, Survived: 0, Pclass: 3, Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S'},
        {PassengerId: 6, Survived: 0, Pclass: 3, Sex: 'male', Age: 28, SibSp: 0, Parch: 0, Fare: 8.46, Embarked: 'Q'},
        {PassengerId: 7, Survived: 0, Pclass: 1, Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Fare: 51.86, Embarked: 'S'},
        {PassengerId: 8, Survived: 0, Pclass: 3, Sex: 'male', Age: 2, SibSp: 3, Parch: 1, Fare: 21.08, Embarked: 'S'},
        {PassengerId: 9, Survived: 1, Pclass: 3, Sex: 'female', Age: 27, SibSp: 0, Parch: 2, Fare: 11.13, Embarked: 'S'},
        {PassengerId: 10, Survived: 1, Pclass: 2, Sex: 'female', Age: 14, SibSp: 1, Parch: 0, Fare: 30.07, Embarked: 'C'}
    ];
    
    appState.rawTestData = [
        {PassengerId: 11, Pclass: 3, Sex: 'male', Age: 4, SibSp: 1, Parch: 1, Fare: 16.70, Embarked: 'S'},
        {PassengerId: 12, Pclass: 1, Sex: 'female', Age: 58, SibSp: 0, Parch: 0, Fare: 26.55, Embarked: 'C'},
        {PassengerId: 13, Pclass: 3, Sex: 'male', Age: 20, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S'},
        {PassengerId: 14, Pclass: 2, Sex: 'male', Age: 39, SibSp: 1, Parch: 5, Fare: 31.28, Embarked: 'S'},
        {PassengerId: 15, Pclass: 3, Sex: 'female', Age: 14, SibSp: 0, Parch: 0, Fare: 7.85, Embarked: 'Q'}
    ];
    
    return true;
}

// ============================================================================
// DATA INSPECTION AND VISUALIZATION
// ============================================================================

function inspectData() {
    if (!appState.rawTrainData || appState.rawTrainData.length === 0) {
        updateStatus('inspectionContent', 'No data loaded', 'error');
        return;
    }
    
    const data = appState.rawTrainData;
    const columns = Object.keys(data[0]);
    const totalRows = data.length;
    
    // Calculate missing values
    let missingHTML = '<h3>Missing Values Analysis</h3><table>';
    missingHTML += '<tr><th>Column</th><th>Missing</th><th>Percentage</th></tr>';
    
    columns.forEach(col => {
        const missingCount = data.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        
        const percentage = ((missingCount / totalRows) * 100).toFixed(1);
        missingHTML += `
            <tr>
                <td>${col}</td>
                <td>${missingCount}</td>
                <td>${percentage}%</td>
            </tr>
        `;
    });
    missingHTML += '</table>';
    
    // Create data preview
    let previewHTML = '<h3>Data Preview (First 5 Rows)</h3><div class="data-preview"><table>';
    previewHTML += '<tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
    
    data.slice(0, 5).forEach(row => {
        previewHTML += '<tr>' + columns.map(col => {
            const val = row[col];
            if (val === null || val === undefined) return '<td>N/A</td>';
            if (typeof val === 'number') return `<td>${val.toFixed(2)}</td>`;
            return `<td>${val}</td>`;
        }).join('') + '</tr>';
    });
    previewHTML += '</table></div>';
    
    // Update inspection content
    document.getElementById('inspectionContent').innerHTML = `
        <div class="metric-box">
            <div class="metric-label">Dataset Shape:</div>
            <div class="metric-value">${totalRows} rows × ${columns.length} columns</div>
        </div>
        ${previewHTML}
        ${missingHTML}
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
        
        if (sex && survivalBySex[sex]) {
            survivalBySex[sex].total++;
            if (survived === 1) survivalBySex[sex].survived++;
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
// DATA PREPROCESSING
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
    
    // Update UI
    document.getElementById('trainSamples').textContent = processedTrain.features.length;
    document.getElementById('valSamples').textContent = Math.floor(processedTrain.features.length * 0.2);
    
    // Show feature information
    document.getElementById('featureList').innerHTML = `
        <h3>Processed Features (${appState.featureNames.length})</h3>
        <p><small>${appState.featureNames.join(', ')}</small></p>
        <div class="metric-box">
            <div class="metric-label">Feature Tensor Shape:</div>
            <div class="metric-value">[${processedTrain.features.length}, ${appState.featureNames.length}]</div>
        </div>
    `;
    
    updateStatus('preprocessStatus', 
        `Preprocessed ${processedTrain.features.length} samples with ${appState.featureNames.length} features`, 
        'success');
    
    return processedTrain;
}

function calculateFeatureStatistics() {
    const stats = {};
    const data = appState.rawTrainData;
    
    // Numerical features
    DATA_SCHEMA.numericalFeatures.forEach(feature => {
        const values = data.map(row => row[feature])
            .filter(val => val !== null && !isNaN(val) && val !== '');
        
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const std = Math.sqrt(values.reduce((sq, val) => sq + Math.pow(val - mean, 2), 0) / values.length);
            const sorted = [...values].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            
            stats[feature] = { mean, std, median, type: 'numerical' };
        } else {
            stats[feature] = { mean: 0, std: 1, median: 0, type: 'numerical' };
        }
    });
    
    // Categorical features
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
            stats[feature] = { 
                mode: 'Unknown', 
                frequencies: {}, 
                categories: [],
                type: 'categorical' 
            };
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
        
        if (isTraining) {
            labels.push(row[DATA_SCHEMA.targetColumn] || 0);
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
    
    // Process numerical features
    DATA_SCHEMA.numericalFeatures.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '' || isNaN(value)) {
            value = appState.featureStats[feature]?.median || 0;
        }
        
        // Standardize
        const stats = appState.featureStats[feature];
        if (stats && stats.std > 0) {
            value = (value - stats.mean) / stats.std;
        }
        
        featureValues.push(value);
        featureNames.push(feature);
    });
    
    // Process categorical features
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '') {
            value = appState.featureStats[feature]?.mode || 'Unknown';
        }
        
        // Simple encoding for categorical features
        if (feature === 'Sex') {
            // One-hot encoding for Sex
            featureValues.push(value === 'female' ? 1 : 0);
            featureValues.push(value === 'male' ? 1 : 0);
            featureNames.push('Sex_female');
            featureNames.push('Sex_male');
        } else if (feature === 'Pclass') {
            // One-hot encoding for Pclass
            [1, 2, 3].forEach(cls => {
                featureValues.push(cls === value ? 1 : 0);
                featureNames.push(`Pclass_${cls}`);
            });
        } else if (feature === 'Embarked') {
            // One-hot encoding for Embarked
            ['C', 'Q', 'S'].forEach(port => {
                featureValues.push(port === value ? 1 : 0);
                featureNames.push(`Embarked_${port}`);
            });
        }
    });
    
    // Add derived features if enabled
    if (document.getElementById('addFamilyFeatures').checked) {
        Object.entries(DATA_SCHEMA.derivedFeatures).forEach(([name, func]) => {
            const value = func(row);
            // Simple normalization for derived features
            const normalizedValue = name === 'FamilySize' ? value / 10 : value;
            featureValues.push(normalizedValue);
            featureNames.push(name);
        });
    }
    
    return { featureValues, featureNames };
}

// ============================================================================
// MODEL CREATION - FIXED VERSION
// ============================================================================

function createModel(inputShape) {
    console.log('Creating model with', inputShape, 'input features');
    
    const model = tf.sequential();
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        // CORRECTED: Feature gate layer with proper architecture
        // This layer learns feature importance weights
        model.add(tf.layers.dense({
            units: inputShape, // Same as input for element-wise multiplication
            activation: 'sigmoid', // Values between 0-1
            useBias: false, // No bias for gate
            kernelInitializer: 'ones', // Start with all ones
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
// TRAINING FUNCTION - FIXED VERSION
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
            xVal: tf.clone(xVal), // Clone to avoid disposal issues
            yVal: tf.clone(yVal)
        };
        
        // Create model
        updateStatus('trainingStatus', 'Creating model...', 'info');
        appState.model = createModel(numFeatures);
        
        // Train model
        updateStatus('trainingStatus', 'Training started (30 epochs)...', 'info');
        
        const history = await appState.model.fit(xTrain, yTrain, {
            epochs: 30,
            batchSize: 32,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const currentEpoch = epoch + 1;
                    const progress = Math.round((currentEpoch / 30) * 100);
                    
                    updateStatus('trainingStatus', 
                        `Epoch ${currentEpoch}/30 (${progress}%) - ` +
                        `Loss: ${logs.loss.toFixed(4)}, ` +
                        `Accuracy: ${(logs.acc * 100).toFixed(1)}%, ` +
                        `Val Loss: ${logs.val_loss.toFixed(4)}`,
                        'success');
                    
                    // Store history
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
        
        console.log('Training completed successfully');
        
        // Clean up tensors (but keep validation data clones)
        featuresTensor.dispose();
        labelsTensor.dispose();
        xTrain.dispose();
        yTrain.dispose();
        xVal.dispose();
        yVal.dispose();
        
        updateStatus('trainingStatus', 
            'Training completed successfully! Click "Evaluate Model" to see metrics.', 
            'success');
        
        // Enable evaluation and export buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('exportModelBtn').disabled = false;
        document.getElementById('predictBtn').disabled = false;
        
        return history;
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('trainingStatus', `Training failed: ${error.message}`, 'error');
        
        // Clean up any tensors that might have been created
        tf.disposeVariables();
    }
}

// ============================================================================
// FEATURE IMPORTANCE
// ============================================================================

function displayFeatureImportance() {
    if (!appState.model) {
        document.getElementById('featureImportance').innerHTML = `
            <h3>Feature Importance</h3>
            <div class="status">
                <i class="fas fa-info-circle"></i> No model available. Please train a model first.
            </div>
        `;
        return;
    }
    
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    const container = document.getElementById('featureImportance');
    
    if (!useFeatureGate) {
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <div class="status">
                <i class="fas fa-info-circle"></i> Feature gate is disabled. 
                Enable "Feature importance learning" in Preprocessing section.
            </div>
        `;
        return;
    }
    
    try {
        // Find feature gate layer
        const gateLayer = appState.model.layers.find(layer => layer.name === 'feature_gate');
        
        if (!gateLayer) {
            container.innerHTML = `
                <h3>Feature Importance</h3>
                <div class="status error">
                    <i class="fas fa-exclamation-triangle"></i> 
                    Feature gate layer not found. The model may have different architecture.
                </div>
            `;
            return;
        }
        
        // Get weights
        const weights = gateLayer.getWeights();
        if (!weights || weights.length === 0) {
            container.innerHTML = `
                <h3>Feature Importance</h3>
                <div class="status error">
                    <i class="fas fa-exclamation-triangle"></i> 
                    No weights found in feature gate layer.
                </div>
            `;
            return;
        }
        
        // Get kernel weights
        const kernel = weights[0];
        const kernelData = Array.from(kernel.dataSync());
        
        console.log('Feature gate kernel shape:', kernel.shape);
        
        // For a diagonal gate, extract diagonal elements
        let importanceValues = [];
        const numFeatures = Math.min(kernel.shape[0], appState.featureNames.length);
        
        if (kernel.shape[0] === kernel.shape[1]) {
            // Square matrix - extract diagonal
            for (let i = 0; i < numFeatures; i++) {
                if (i < kernel.shape[0] && i < kernel.shape[1]) {
                    importanceValues.push(kernelData[i * kernel.shape[1] + i]);
                }
            }
        } else {
            // Not square - take mean of columns
            for (let i = 0; i < numFeatures; i++) {
                let sum = 0;
                for (let j = 0; j < kernel.shape[0]; j++) {
                    if (i < kernel.shape[1]) {
                        sum += kernelData[j * kernel.shape[1] + i];
                    }
                }
                importanceValues.push(sum / kernel.shape[0]);
            }
        }
        
        // Fill any missing values
        while (importanceValues.length < appState.featureNames.length) {
            importanceValues.push(0);
        }
        
        // Apply sigmoid activation (since gate uses sigmoid)
        const sigmoid = (x) => 1 / (1 + Math.exp(-x));
        const activatedValues = importanceValues.map(sigmoid);
        
        // Normalize for display (0 to 1)
        const maxVal = Math.max(...activatedValues);
        const minVal = Math.min(...activatedValues);
        const normalizedValues = activatedValues.map(val => 
            maxVal !== minVal ? (val - minVal) / (maxVal - minVal) : 0.5
        );
        
        // Create feature objects
        const features = appState.featureNames.map((name, idx) => ({
            name: name || `Feature_${idx + 1}`,
            importance: normalizedValues[idx] || 0,
            rawValue: importanceValues[idx] || 0,
            activatedValue: activatedValues[idx] || 0
        }));
        
        // Sort by importance
        features.sort((a, b) => b.importance - a.importance);
        
        // Display results
        let html = '<h3>Feature Importance Analysis</h3>';
        html += '<p>Importance scores from learned sigmoid gate (normalized 0-1):</p>';
        
        if (features.length === 0) {
            html += '<p class="status">No feature importance data available.</p>';
            container.innerHTML = html;
            return;
        }
        
        // Create table
        html += `
            <div style="overflow-x: auto; margin: 15px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="background: rgba(68, 138, 255, 0.2);">
                            <th style="padding: 8px; text-align: left;">Rank</th>
                            <th style="padding: 8px; text-align: left;">Feature</th>
                            <th style="padding: 8px; text-align: left;">Score</th>
                            <th style="padding: 8px; text-align: left;">Importance</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        const topFeatures = features.slice(0, Math.min(15, features.length));
        topFeatures.forEach((feature, idx) => {
            const percentage = Math.round(feature.importance * 100);
            const barWidth = Math.max(30, Math.min(150, percentage * 1.5));
            
            html += `
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                    <td style="padding: 8px; text-align: center; font-weight: bold;">${idx + 1}</td>
                    <td style="padding: 8px;">${feature.name}</td>
                    <td style="padding: 8px; font-family: monospace; color: #82b1ff;">${feature.importance.toFixed(3)}</td>
                    <td style="padding: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: ${barWidth}px; height: 8px; 
                                 background: linear-gradient(90deg, #448aff, #82b1ff); 
                                 border-radius: 4px;"></div>
                            <span style="font-size: 0.85em;">${percentage}%</span>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        html += '</tbody></table></div>';
        
        // Statistics
        if (features.length > 0) {
            const avgImportance = features.reduce((sum, f) => sum + f.importance, 0) / features.length;
            const mostImportant = features[0];
            const leastImportant = features[features.length - 1];
            
            html += `
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 15px;">
                    <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                        <div style="font-size: 0.85em; color: #bbdefb;">Most Important</div>
                        <div style="font-weight: bold; font-size: 0.9em;">${mostImportant.name}</div>
                        <div style="color: #82b1ff; font-size: 0.9em;">${mostImportant.importance.toFixed(3)}</div>
                    </div>
                    <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                        <div style="font-size: 0.85em; color: #bbdefb;">Average</div>
                        <div style="font-weight: bold; font-size: 0.9em;">${avgImportance.toFixed(3)}</div>
                        <div style="color: #bbdefb; font-size: 0.8em;">/ 1.0</div>
                    </div>
                    <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                        <div style="font-size: 0.85em; color: #bbdefb;">Features</div>
                        <div style="font-weight: bold; font-size: 0.9em;">${features.length}</div>
                        <div style="color: #bbdefb; font-size: 0.8em;">analyzed</div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <div class="status error">
                <i class="fas fa-exclamation-triangle"></i> 
                Error: ${error.message}
            </div>
        `;
    }
}

// ============================================================================
// EVALUATION AND METRICS
// ============================================================================

function calculateAUC(predictions, trueLabels) {
    if (!predictions || !trueLabels || predictions.length === 0 || predictions.length !== trueLabels.length) {
        console.warn('Invalid data for AUC calculation');
        return 0.5;
    }
    
    // Create pairs
    const pairs = predictions.map((p, i) => ({ score: p, label: trueLabels[i] }));
    
    // Sort by score (descending)
    pairs.sort((a, b) => b.score - a.score);
    
    const totalPositives = pairs.filter(p => p.label === 1).length;
    const totalNegatives = pairs.filter(p => p.label === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) {
        console.warn('Cannot calculate AUC: need both positive and negative samples');
        return 0.5;
    }
    
    // Calculate ROC points
    const thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
    let auc = 0;
    let prevFPR = 0;
    let prevTPR = 0;
    
    for (const threshold of thresholds) {
        let tp = 0, fp = 0;
        
        for (const pair of pairs) {
            if (pair.score >= threshold) {
                if (pair.label === 1) tp++;
                else fp++;
            }
        }
        
        const tpr = tp / totalPositives;
        const fpr = fp / totalNegatives;
        
        // Calculate trapezoid area
        if (threshold < 1.0) {
            auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
        }
        
        prevFPR = fpr;
        prevTPR = tpr;
    }
    
    // Ensure AUC is between 0 and 1
    return Math.max(0, Math.min(1, auc));
}

function updateMetrics(threshold, predictions, trueLabels) {
    if (!predictions || !trueLabels) {
        if (!appState.validationData) return;
        
        try {
            const preds = appState.model.predict(appState.validationData.xVal);
            predictions = Array.from(preds.dataSync());
            trueLabels = Array.from(appState.validationData.yVal.dataSync());
            preds.dispose();
        } catch (error) {
            console.error('Error getting predictions:', error);
            return;
        }
    }
    
    // Validate data
    if (predictions.length !== trueLabels.length) {
        console.error('Predictions and labels length mismatch');
        return;
    }
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    predictions.forEach((pred, idx) => {
        const predicted = pred >= threshold ? 1 : 0;
        const actual = trueLabels[idx];
        
        if (actual === 1 && predicted === 1) tp++;
        else if (actual === 0 && predicted === 1) fp++;
        else if (actual === 0 && predicted === 0) tn++;
        else if (actual === 1 && predicted === 0) fn++;
    });
    
    // Update confusion matrix display
    const matrixHTML = `
        <div></div>
        <div class="confusion-cell" style="background: rgba(68, 138, 255, 0.2);">Predicted 0</div>
        <div class="confusion-cell" style="background: rgba(68, 138, 255, 0.2);">Predicted 1</div>
        <div class="confusion-cell" style="background: rgba(68, 138, 255, 0.2);">Actual 0</div>
        <div class="confusion-cell true-negative">${tn}</div>
        <div class="confusion-cell false-positive">${fp}</div>
        <div class="confusion-cell" style="background: rgba(68, 138, 255, 0.2);">Actual 1</div>
        <div class="confusion-cell false-negative">${fn}</div>
        <div class="confusion-cell true-positive">${tp}</div>
    `;
    
    document.getElementById('confusionMatrix').innerHTML = matrixHTML;
    document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
    
    // Calculate metrics with safe division
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1Score').textContent = f1.toFixed(3);
}

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
        
        console.log('Evaluation predictions sample:', {
            preds: predValues.slice(0, 5),
            truths: trueValues.slice(0, 5)
        });
        
        // Calculate AUC
        const auc = calculateAUC(predValues, trueValues);
        document.getElementById('aucScore').textContent = auc.toFixed(3);
        
        // Update metrics with threshold 0.5
        updateMetrics(0.5, predValues, trueValues);
        
        // Display feature importance if gate is enabled
        const useFeatureGate = document.getElementById('useFeatureGate').checked;
        if (useFeatureGate) {
            setTimeout(() => {
                displayFeatureImportance();
            }, 100);
        }
        
        updateStatus('loadStatus', `Evaluation completed. AUC: ${auc.toFixed(3)}`, 'success');
        
        predictions.dispose();
        
    } catch (error) {
        updateStatus('loadStatus', `Evaluation failed: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
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
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
        let sampleHTML = '<h3>Sample Predictions</h3><table>';
        sampleHTML += '<tr><th>PassengerId</th><th>Probability</th><th>Predicted</th></tr>';
        
        const sampleSize = Math.min(5, probabilities.length);
        for (let i = 0; i < sampleSize; i++) {
            const predClass = probabilities[i] >= threshold ? 1 : 0;
            sampleHTML += `
                <tr>
                    <td>${appState.testPassengerIds[i]}</td>
                    <td>${probabilities[i].toFixed(4)}</td>
                    <td>${predClass}</td>
                </tr>
            `;
        }
        
        sampleHTML += '</table>';
        
        updateStatus('predictionStatus', 
            `Generated ${probabilities.length} predictions. ${sampleHTML}`, 
            'success');
        
        // Enable export buttons
        document.getElementById('exportSubmissionBtn').disabled = false;
        document.getElementById('exportProbabilitiesBtn').disabled = false;
        
        testFeatures.dispose();
        predictions.dispose();
        
    } catch (error) {
        updateStatus('predictionStatus', `Prediction failed: ${error.message}`, 'error');
    }
}

function exportSubmissionCSV() {
    if (!appState.testPredictions || !appState.testPassengerIds) {
        alert('Please generate predictions first');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
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

async function exportModel() {
    if (!appState.model) {
        alert('Please train a model first');
        return;
    }
    
    try {
        await appState.model.save('downloads://titanic-model');
        updateStatus('trainingStatus', 'Model downloaded successfully!', 'success');
    } catch (error) {
        updateStatus('trainingStatus', `Failed to export model: ${error.message}`, 'error');
    }
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
        updateMetrics(threshold);
    });
    
    // Export buttons
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmissionCSV);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilitiesCSV);
    document.getElementById('exportModelBtn').addEventListener('click', exportModel);
    
    // File input changes
    document.getElementById('trainFile').addEventListener('change', () => {
        document.getElementById('useDemoData').checked = false;
    });
    
    document.getElementById('testFile').addEventListener('change', () => {
        document.getElementById('useDemoData').checked = false;
    });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateStatus('loadStatus', 'Ready to load Titanic dataset. Check "Use demo data" for instant demo.', 'info');
});
