// app.js - Complete Titanic Classifier with File Upload Support

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
// CSV FILE PARSING - FIXED VERSION
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
                
                // Handle different line endings and quotes
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
                    
                    // Match values to headers
                    const row = {};
                    headers.forEach((header, index) => {
                        let value = values[index] || '';
                        value = value.trim();
                        
                        // Convert numeric values
                        if (!isNaN(value) && value !== '') {
                            value = parseFloat(value);
                        } else if (value === '') {
                            value = null; // Treat empty as null
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

function parseCSVLine(line) {
    const values = [];
    let currentValue = '';
    let insideQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            if (insideQuotes && nextChar === '"') {
                // Escaped quote
                currentValue += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted section
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            // End of value
            values.push(currentValue);
            currentValue = '';
        } else {
            currentValue += char;
        }
    }
    
    // Add the last value
    values.push(currentValue);
    
    return values;
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
            
            stats[feature] = { mean, std, median };
        } else {
            stats[feature] = { mean: 0, std: 1, median: 0 };
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
                freq[val] = (freq[val] || 0) + 1;
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
            
            stats[feature] = { mode, frequencies: freq };
        } else {
            stats[feature] = { mode: 'Unknown', frequencies: {} };
        }
    });
    
    appState.featureStats = stats;
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
    
    // Process categorical features (one-hot encoding)
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '') {
            value = appState.featureStats[feature]?.mode || 'Unknown';
        }
        
        // Get unique categories
        const categories = Object.keys(appState.featureStats[feature]?.frequencies || {});
        
        if (categories.length > 0) {
            // Create one-hot encoding
            categories.forEach(category => {
                featureValues.push(category === value ? 1 : 0);
                featureNames.push(`${feature}_${category}`);
            });
        } else {
            // Simple binary encoding for known values
            if (feature === 'Sex') {
                featureValues.push(value === 'female' ? 1 : 0);
                featureNames.push('Sex_female');
            } else if (feature === 'Embarked') {
                const encoded = value === 'C' ? 0 : value === 'Q' ? 0.5 : 1;
                featureValues.push(encoded);
                featureNames.push('Embarked_encoded');
            }
        }
    });
    
    // Add derived features if enabled
    if (document.getElementById('addFamilyFeatures').checked) {
        Object.entries(DATA_SCHEMA.derivedFeatures).forEach(([name, func]) => {
            const value = func(row);
            featureValues.push(value);
            featureNames.push(name);
        });
    }
    
    return { featureValues, featureNames };
}

// ============================================================================
// MODEL CREATION AND TRAINING
// ============================================================================

function createModel(inputShape) {
    const model = tf.sequential();
    
    // Optional feature gate layer
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        // Feature gate layer must have inputShape
        model.add(tf.layers.dense({
            units: inputShape,
            activation: 'sigmoid',
            useBias: false,
            kernelInitializer: 'ones',
            trainable: true,
            inputShape: [inputShape],
            name: 'feature_gate'
        }));
        
        // Main model architecture
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            name: 'hidden'
        }));
    } else {
        // Without feature gate, first layer needs inputShape
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [inputShape],
            kernelInitializer: 'heNormal',
            name: 'hidden'
        }));
    }
    
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
    
    // Store model reference in app state
    appState.model = model;
    
    // Display model summary
    document.getElementById('modelSummary').innerHTML = `
        <h3>Model Architecture</h3>
        <p>Input shape: [${inputShape}]</p>
        <p>Total parameters: ${model.countParams().toLocaleString()}</p>
        <p><small>${useFeatureGate ? 'With' : 'Without'} feature importance gate</small></p>
        <p><small>Layers: ${model.layers.map(l => l.name).join(' → ')}</small></p>
    `;
    
    return model;
}

async function trainModel() {
    try {
        // Preprocess data first
        const processedData = preprocessData();
        
        // Check if we have features
        if (!processedData.features || processedData.features.length === 0) {
            throw new Error('No features available after preprocessing');
        }
        
        // Create tensors
        const featuresTensor = tf.tensor2d(processedData.features);
        const labelsTensor = tf.tensor2d(processedData.labels, [processedData.labels.length, 1]);
        
        console.log(`Feature tensor shape: ${featuresTensor.shape}`);
        console.log(`Label tensor shape: ${labelsTensor.shape}`);
        
        // Create validation split
        const splitIndex = Math.floor(featuresTensor.shape[0] * 0.8);
        
        const xTrain = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const yTrain = labelsTensor.slice([0, 0], [splitIndex, -1]);
        const xVal = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const yVal = labelsTensor.slice([splitIndex, 0], [-1, -1]);
        
        appState.validationData = { xVal, yVal };
        
        // Create and train model - pass correct input shape
        appState.model = createModel(featuresTensor.shape[1]);
        
        updateStatus('trainingStatus', 'Training started...', 'info');
        
        const history = await appState.model.fit(xTrain, yTrain, {
            epochs: 30,
            batchSize: 32,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = Math.round((epoch + 1) / 30 * 100);
                    updateStatus('trainingStatus', 
                        `Epoch ${epoch + 1}/30 (${progress}%) - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}`, 
                        'success');
                    
                    appState.trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        val_loss: logs.val_loss,
                        acc: logs.acc,
                        val_acc: logs.val_acc
                    });
                },
                onTrainEnd: () => {
                    // Display feature importance after training
                    setTimeout(() => {
                        displayFeatureImportance();
                    }, 100);
                }
            }
        });
        
        // Cleanup tensors
        featuresTensor.dispose();
        labelsTensor.dispose();
        xTrain.dispose();
        yTrain.dispose();
        
        updateStatus('trainingStatus', 'Training completed successfully!', 'success');
        
        // Enable evaluation and export buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('exportModelBtn').disabled = false;
        
        return history;
        
    } catch (error) {
        updateStatus('trainingStatus', `Training failed: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

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
    
    // Debug logging
    console.log('Processed training data:', {
        numSamples: processedTrain.features.length,
        numFeatures: processedTrain.features[0] ? processedTrain.features[0].length : 0,
        featureNames: processedTrain.featureNames,
        sampleFeatures: processedTrain.features[0]
    });
    
    // Update UI
    document.getElementById('trainSamples').textContent = processedTrain.features.length;
    document.getElementById('valSamples').textContent = Math.floor(processedTrain.features.length * 0.2);
    
    // Show feature information
    document.getElementById('featureList').innerHTML = `
        <h3>Processed Features (${processedTrain.featureNames.length})</h3>
        <p><small>${processedTrain.featureNames.join(', ')}</small></p>
        <div class="metric-box">
            <div class="metric-label">Feature Tensor Shape:</div>
            <div class="metric-value">[${processedTrain.features.length}, ${processedTrain.featureNames.length}]</div>
        </div>
    `;
    
    updateStatus('preprocessStatus', 
        `Preprocessed ${processedTrain.features.length} samples with ${processedTrain.featureNames.length} features`, 
        'success');
    
    return processedTrain;
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
        
        // Get predictions
        const predictions = appState.model.predict(xVal);
        const predValues = Array.from(predictions.dataSync());
        const trueValues = Array.from(yVal.dataSync());
        
        // Calculate initial metrics with threshold 0.5
        updateMetrics(0.5, predValues, trueValues);
        
        // Calculate AUC (simplified for demo)
        const auc = calculateAUC(predValues, trueValues);
        document.getElementById('aucScore').textContent = auc.toFixed(3);
        
        // Display feature importance if gate is enabled
        displayFeatureImportance();
        
        updateStatus('loadStatus', 'Evaluation completed. Adjust threshold slider.', 'success');
        
        // Enable prediction button
        document.getElementById('predictBtn').disabled = false;
        
        predictions.dispose();
        
    } catch (error) {
        updateStatus('loadStatus', `Evaluation failed: ${error.message}`, 'error');
    }
}

function calculateAUC(predictions, trueLabels) {
    // Simplified AUC calculation for demo
    const sorted = predictions.map((p, i) => ({ p, y: trueLabels[i] }))
        .sort((a, b) => a.p - b.p);
    
    let auc = 0;
    let prevFalseRate = 0;
    let prevTrueRate = 0;
    let totalPositives = sorted.filter(d => d.y === 1).length;
    let totalNegatives = sorted.filter(d => d.y === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) return 0.5;
    
    for (let i = 0; i <= 100; i++) {
        const threshold = i / 100;
        
        let tp = 0, fp = 0;
        sorted.forEach(d => {
            if (d.p >= threshold) {
                if (d.y === 1) tp++;
                else fp++;
            }
        });
        
        const tpr = tp / totalPositives;
        const fpr = fp / totalNegatives;
        
        if (i > 0) {
            auc += (fpr - prevFalseRate) * (tpr + prevTrueRate) / 2;
        }
        
        prevFalseRate = fpr;
        prevTrueRate = tpr;
    }
    
    return auc;
}

function updateMetrics(threshold, predictions, trueLabels) {
    if (!predictions || !trueLabels) {
        if (!appState.validationData) return;
        
        const preds = appState.model.predict(appState.validationData.xVal);
        predictions = Array.from(preds.dataSync());
        trueLabels = Array.from(appState.validationData.yVal.dataSync());
        preds.dispose();
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
    
    // Calculate metrics
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1Score').textContent = f1.toFixed(3);
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
// FEATURE IMPORTANCE EXTRACTION AND DISPLAY
// ============================================================================

function displayFeatureImportance() {
    if (!appState.model) {
        return;
    }
    
    const container = document.getElementById('featureImportance');
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (!useFeatureGate) {
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <p>Feature importance gate is disabled. Enable it in Preprocessing section.</p>
        `;
        return;
    }
    
    try {
        // Get the feature gate layer
        const gateLayer = appState.model.layers.find(layer => layer.name === 'feature_gate');
        
        if (!gateLayer) {
            container.innerHTML = `
                <h3>Feature Importance</h3>
                <p>No feature gate layer found in the model.</p>
            `;
            return;
        }
        
        // Get weights from the gate layer
        const weights = gateLayer.getWeights()[0];
        const importanceValues = Array.from(weights.dataSync());
        
        // Normalize importance scores (0 to 1)
        const maxVal = Math.max(...importanceValues);
        const minVal = Math.min(...importanceValues);
        const normalized = importanceValues.map(val => 
            maxVal !== minVal ? (val - minVal) / (maxVal - minVal) : 0.5
        );
        
        // Create feature importance display
        let html = '<h3>Feature Importance (Sigmoid Gate)</h3>';
        html += '<p>Higher values indicate more important features for prediction:</p>';
        
        // Create bars for each feature
        normalized.forEach((importance, index) => {
            const featureName = appState.featureNames[index] || `Feature ${index + 1}`;
            const rawValue = importanceValues[index].toFixed(3);
            const percentage = Math.round(importance * 100);
            
            html += `
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 0.9em;">${featureName}</span>
                        <span style="font-weight: bold; color: #82b1ff;">${rawValue}</span>
                    </div>
                    <div style="height: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 5px; overflow: hidden;">
                        <div style="height: 100%; width: ${percentage}%; 
                            background: linear-gradient(90deg, #448aff, #82b1ff); 
                            border-radius: 5px;"></div>
                    </div>
                </div>
            `;
        });
        
        // Add summary statistics
        const avgImportance = (normalized.reduce((a, b) => a + b, 0) / normalized.length).toFixed(3);
        
        html += `
            <div style="margin-top: 15px; padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Average importance:</span>
                    <span style="font-weight: bold;">${avgImportance}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                    <span>Most important feature:</span>
                    <span style="font-weight: bold;">${appState.featureNames[normalized.indexOf(Math.max(...normalized))]}</span>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error displaying feature importance:', error);
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <p class="status error">Error displaying feature importance: ${error.message}</p>
        `;
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
