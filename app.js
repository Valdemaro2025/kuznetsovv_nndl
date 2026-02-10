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
    
   // Debug logging
    console.log('=== PREPROCESSING DEBUG INFO ===');
    console.log('Training samples:', processedTrain.features.length);
    console.log('Feature count per sample:', processedTrain.features[0]?.length || 0);
    console.log('Feature names:', processedTrain.featureNames);
    console.log('First sample features:', processedTrain.features[0]);
    console.log('Feature names count:', processedTrain.featureNames.length);
    console.log('===============================');
    
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
    
    // Categorical features - FIXED to collect all categories
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        const values = data.map(row => row[feature])
            .filter(val => val !== null && val !== '');
        
        if (values.length > 0) {
            // Count frequencies for ALL values
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
            
            console.log(`Categories for ${feature}:`, Object.keys(freq));
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
    console.log('Feature statistics:', stats);
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
    
    // Process categorical features - FIXED
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '') {
            value = appState.featureStats[feature]?.mode || 'Unknown';
        }
        
        // Get unique categories from training data stats
        const categories = Object.keys(appState.featureStats[feature]?.frequencies || {});
        
        if (categories.length > 0) {
            // One-hot encoding for up to 5 most common categories
            const topCategories = categories
                .sort((a, b) => appState.featureStats[feature].frequencies[b] - appState.featureStats[feature].frequencies[a])
                .slice(0, 5);
            
            topCategories.forEach(category => {
                featureValues.push(category === value ? 1 : 0);
                featureNames.push(`${feature}_${category}`);
            });
        } else {
            // Fallback for test data or unknown categories
            if (feature === 'Sex') {
                featureValues.push(value === 'female' ? 1 : 0);
                featureNames.push('Sex_female');
            } else if (feature === 'Pclass') {
                // One-hot for Pclass (1, 2, 3)
                [1, 2, 3].forEach(cls => {
                    featureValues.push(cls === value ? 1 : 0);
                    featureNames.push(`Pclass_${cls}`);
                });
            } else if (feature === 'Embarked') {
                // One-hot for Embarked (C, Q, S)
                ['C', 'Q', 'S'].forEach(port => {
                    featureValues.push(port === value ? 1 : 0);
                    featureNames.push(`Embarked_${port}`);
                });
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
    console.log('Creating model with input shape:', inputShape);
    
    const model = tf.sequential();
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        console.log('Creating model WITH feature gate');
        // Feature gate should have inputShape units
        model.add(tf.layers.dense({
            units: inputShape, // Количество единиц должно совпадать с количеством признаков
            activation: 'sigmoid',
            useBias: false,
            kernelInitializer: 'ones',
            trainable: true,
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
        console.log('Creating model WITHOUT feature gate');
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
    
    console.log('Model created successfully');
    console.log('Model layers:', model.layers.map(l => `${l.name}: ${l.outputShape}`));
    
    // Display model summary
    document.getElementById('modelSummary').innerHTML = `
        <h3>Model Architecture</h3>
        <p>Input features: ${inputShape}</p>
        <p>Hidden units: 16</p>
        <p>Feature gate: ${useFeatureGate ? 'Enabled' : 'Disabled'}</p>
        <p>Total parameters: ${model.countParams().toLocaleString()}</p>
    `;
    
    return model;
}

async function trainModel() {
    try {
        // Preprocess data first
        const processedData = preprocessData();
        
        // Validate data
        if (!processedData.features || processedData.features.length === 0) {
            throw new Error('No features available after preprocessing');
        }
        
        const numFeatures = processedData.features[0].length;
        console.log(`Training with ${processedData.features.length} samples, ${numFeatures} features`);
        
        // Create tensors
        const featuresTensor = tf.tensor2d(processedData.features);
        const labelsTensor = tf.tensor2d(processedData.labels, [processedData.labels.length, 1]);
        
        console.log('Feature tensor shape:', featuresTensor.shape);
        console.log('Label tensor shape:', labelsTensor.shape);
        
        // Create validation split
        const splitIndex = Math.floor(featuresTensor.shape[0] * 0.8);
        
        const xTrain = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const yTrain = labelsTensor.slice([0, 0], [splitIndex, -1]);
        const xVal = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const yVal = labelsTensor.slice([splitIndex, 0], [-1, -1]);
        
        appState.validationData = { xVal, yVal };
        
        // Create model with correct input shape
        appState.model = createModel(numFeatures);
        
        updateStatus('trainingStatus', 'Training started...', 'info');
        
        // Train model
        await appState.model.fit(xTrain, yTrain, {
            epochs: 30,
            batchSize: 32,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = Math.round((epoch + 1) / 30 * 100);
                    updateStatus('trainingStatus', 
                        `Epoch ${epoch + 1}/30 - Loss: ${logs.loss.toFixed(4)}, Acc: ${(logs.acc * 100).toFixed(1)}%`, 
                        'success');
                },
                onTrainEnd: () => {
                    console.log('Training completed successfully');
                    console.log('Model summary:', appState.model.summary());
                }
            }
        });
        
        // Cleanup
        [featuresTensor, labelsTensor, xTrain, yTrain].forEach(t => t.dispose());
        
        updateStatus('trainingStatus', 'Training completed! Click "Evaluate Model" for metrics.', 'success');
        
        // Enable buttons
        document.getElementById('evaluateBtn').disabled = false;
        document.getElementById('exportModelBtn').disabled = false;
        
    } catch (error) {
        updateStatus('trainingStatus', `Training failed: ${error.message}`, 'error');
        console.error('Training error details:', error);
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
        
        console.log('Evaluation predictions:', {
            count: predValues.length,
            sample: predValues.slice(0, 5),
            trueSample: trueValues.slice(0, 5)
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
            }, 500);
        }
        
        updateStatus('loadStatus', `Evaluation completed. AUC: ${auc.toFixed(3)}`, 'success');
        
        // Enable prediction button
        document.getElementById('predictBtn').disabled = false;
        
        predictions.dispose();
        
    } catch (error) {
        updateStatus('loadStatus', `Evaluation failed: ${error.message}`, 'error');
        console.error('Evaluation error:', error);
    }
}

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
    
    // Calculate true positive rate and false positive rate at different thresholds
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
            <p>Feature importance learning is disabled. Enable it in Preprocessing section.</p>
        `;
        return;
    }
    
    try {
        // Find feature gate layer
        const gateLayer = appState.model.layers.find(layer => layer.name === 'feature_gate');
        
        if (!gateLayer) {
            container.innerHTML = `
                <h3>Feature Importance</h3>
                <p class="status error">Feature gate layer not found in model.</p>
            `;
            return;
        }
        
        // Get weights from gate layer
        const weights = gateLayer.getWeights();
        if (!weights || weights.length === 0) {
            container.innerHTML = `
                <h3>Feature Importance</h3>
                <p class="status error">No weights found in feature gate layer.</p>
            `;
            return;
        }
        
        // The weights are the kernel matrix (inputShape x inputShape)
        // For a diagonal gate, we should look at the diagonal elements
        const kernel = weights[0];
        console.log('Feature gate kernel shape:', kernel.shape);
        
        // Extract diagonal elements
        const importanceValues = [];
        const kernelData = Array.from(kernel.dataSync());
        
        if (kernel.shape[0] === kernel.shape[1]) {
            // Square matrix - extract diagonal
            for (let i = 0; i < kernel.shape[0]; i++) {
                importanceValues.push(kernelData[i * kernel.shape[1] + i]);
            }
        } else {
            // Not square - take first row or flatten
            const minDim = Math.min(kernel.shape[0], kernel.shape[1]);
            for (let i = 0; i < minDim; i++) {
                importanceValues.push(kernelData[i]);
            }
        }
        
        console.log('Importance values (raw):', importanceValues);
        
        // Apply sigmoid activation to get importance scores between 0 and 1
        const sigmoid = (x) => 1 / (1 + Math.exp(-x));
        const activatedImportance = importanceValues.map(sigmoid);
        
        console.log('Importance values (sigmoid):', activatedImportance);
        
        // Match with feature names
        const featureImportance = [];
        const minLength = Math.min(activatedImportance.length, appState.featureNames.length);
        
        for (let i = 0; i < minLength; i++) {
            featureImportance.push({
                name: appState.featureNames[i] || `Feature_${i}`,
                importance: activatedImportance[i],
                rawValue: importanceValues[i],
                rank: i + 1
            });
        }
        
        // Sort by importance (descending)
        featureImportance.sort((a, b) => b.importance - a.importance);
        
        // Update ranks after sorting
        featureImportance.forEach((f, idx) => {
            f.rank = idx + 1;
        });
        
        // Display
        let html = '<h3>Feature Importance (Sigmoid Gate)</h3>';
        html += '<p>Importance scores after sigmoid activation (0-1 scale):</p>';
        
        if (featureImportance.length === 0) {
            html += '<p class="status error">No feature importance data available.</p>';
            container.innerHTML = html;
            return;
        }
        
        // Create table for better readability
        html += `
            <div style="overflow-x: auto; margin: 15px 0;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="background: rgba(68, 138, 255, 0.2);">
                            <th style="padding: 8px; text-align: left; width: 40px;">Rank</th>
                            <th style="padding: 8px; text-align: left;">Feature</th>
                            <th style="padding: 8px; text-align: left; width: 80px;">Score</th>
                            <th style="padding: 8px; text-align: left; width: 120px;">Importance</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        featureImportance.forEach((feature, idx) => {
            const percentage = Math.round(feature.importance * 100);
            const barWidth = Math.max(20, percentage); // Minimum 20px for visibility
            
            html += `
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                    <td style="padding: 8px; text-align: center; font-weight: bold; color: #448aff;">${feature.rank}</td>
                    <td style="padding: 8px;">${feature.name}</td>
                    <td style="padding: 8px; font-family: monospace; color: #82b1ff;">${feature.importance.toFixed(3)}</td>
                    <td style="padding: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: ${barWidth}px; height: 8px; background: linear-gradient(90deg, #448aff, #82b1ff); border-radius: 4px;"></div>
                            <span style="font-size: 0.85em;">${percentage}%</span>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        // Add statistics
        const avgImportance = featureImportance.reduce((sum, f) => sum + f.importance, 0) / featureImportance.length;
        const stdImportance = Math.sqrt(
            featureImportance.reduce((sum, f) => sum + Math.pow(f.importance - avgImportance, 2), 0) / featureImportance.length
        );
        
        html += `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 15px;">
                <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                    <div style="font-size: 0.85em; color: #bbdefb;">Total Features</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #82b1ff;">${featureImportance.length}</div>
                </div>
                <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                    <div style="font-size: 0.85em; color: #bbdefb;">Avg Importance</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #82b1ff;">${avgImportance.toFixed(3)}</div>
                </div>
                <div style="padding: 10px; background: rgba(68, 138, 255, 0.1); border-radius: 6px;">
                    <div style="font-size: 0.85em; color: #bbdefb;">Std Dev</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #82b1ff;">${stdImportance.toFixed(3)}</div>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 12px; background: rgba(105, 240, 174, 0.1); border-radius: 6px; border-left: 4px solid #69f0ae;">
                <div style="font-weight: bold; margin-bottom: 5px;">Interpretation:</div>
                <div style="font-size: 0.9em; color: #bbdefb;">
                    • Scores close to 1: Feature is very important for prediction<br>
                    • Scores close to 0.5: Feature has moderate importance<br>
                    • Scores close to 0: Feature is less important (may be redundant)<br>
                    • All scores are normalized between 0 and 1 using sigmoid function
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        container.innerHTML = `
            <h3>Feature Importance</h3>
            <p class="status error">Error calculating feature importance: ${error.message}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Try retraining the model or disable feature gate.</p>
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
