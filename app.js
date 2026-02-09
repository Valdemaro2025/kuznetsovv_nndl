// app.js
// Titanic Binary Classifier - Complete TensorFlow.js Implementation
// All processing happens in the browser, no backend required

// ============================================================================
// GLOBAL STATE
// ============================================================================
let rawTrainData = null;
let rawTestData = null;
let processedTrainData = null;
let processedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let featureStats = {};
let featureNames = [];
let testPredictions = null;
let testPassengerIds = null;
let featureGateLayer = null;

// ============================================================================
// SCHEMA CONFIGURATION - MODIFY FOR OTHER DATASETS
// ============================================================================
const SCHEMA = {
    // Target variable column name (binary classification)
    TARGET: 'Survived',
    
    // Identifier column (excluded from training)
    ID: 'PassengerId',
    
    // Numerical features (will be standardized)
    NUMERICAL_FEATURES: ['Age', 'Fare', 'SibSp', 'Parch'],
    
    // Categorical features (will be one-hot encoded)
    CATEGORICAL_FEATURES: ['Sex', 'Pclass', 'Embarked'],
    
    // Features to exclude completely
    EXCLUDE: ['Name', 'Ticket', 'Cabin'],
    
    // Derived feature functions (toggleable via UI)
    DERIVED_FEATURES: {
        'FamilySize': (row) => row.SibSp + row.Parch + 1,
        'IsAlone': (row) => (row.SibSp + row.Parch === 0) ? 1 : 0
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function setStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.innerHTML = `<i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i> ${message}`;
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
    link.click();
}

// ============================================================================
// CSV PARSING
// ============================================================================
async function parseCSV(file) {
    return new Promise((resolve, reject) => {
        if (!file) {
            reject(new Error('No file provided'));
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split('\n').filter(line => line.trim() !== '');
                
                if (lines.length < 2) {
                    reject(new Error('CSV file too short'));
                    return;
                }
                
                // Parse headers
                const headers = lines[0].split(',').map(h => h.trim());
                
                // Parse rows with quoted field handling
                const rows = [];
                for (let i = 1; i < lines.length; i++) {
                    let line = lines[i];
                    const values = [];
                    let inQuotes = false;
                    let currentValue = '';
                    
                    for (let char of line) {
                        if (char === '"') {
                            inQuotes = !inQuotes;
                        } else if (char === ',' && !inQuotes) {
                            values.push(currentValue.trim());
                            currentValue = '';
                        } else {
                            currentValue += char;
                        }
                    }
                    values.push(currentValue.trim());
                    
                    // Ensure we have the right number of columns
                    if (values.length === headers.length) {
                        const row = {};
                        headers.forEach((header, idx) => {
                            let value = values[idx];
                            // Convert numeric values
                            if (!isNaN(value) && value.trim() !== '') {
                                value = parseFloat(value);
                            }
                            row[header] = value;
                        });
                        rows.push(row);
                    }
                }
                
                resolve(rows);
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// ============================================================================
// DATA LOADING
// ============================================================================
async function loadDemoData() {
    try {
        // Load Titanic dataset from online sources
        const trainResponse = await fetch('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv');
        const trainText = await trainResponse.text();
        
        // Parse CSV
        const lines = trainText.split('\n').filter(line => line.trim() !== '');
        const headers = lines[0].split(',').map(h => h.trim());
        
        rawTrainData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            const row = {};
            headers.forEach((header, idx) => {
                let value = values[idx] ? values[idx].trim() : '';
                if (!isNaN(value) && value !== '') {
                    value = parseFloat(value);
                }
                row[header] = value;
            });
            rawTrainData.push(row);
        }
        
        // Create test data from last 20% for demo
        const testSize = Math.floor(rawTrainData.length * 0.2);
        rawTestData = rawTrainData.slice(-testSize);
        rawTrainData = rawTrainData.slice(0, -testSize);
        
        setStatus('loadStatus', `Loaded demo data: ${rawTrainData.length} training, ${rawTestData.length} test samples`, 'success');
        return true;
    } catch (error) {
        setStatus('loadStatus', `Failed to load demo data: ${error.message}`, 'error');
        return false;
    }
}

async function loadData() {
    try {
        const trainFile = document.getElementById('trainFile').files[0];
        const testFile = document.getElementById('testFile').files[0];
        const useDemo = document.getElementById('useDemoData').checked;
        
        if (useDemo || (!trainFile && !testFile)) {
            return await loadDemoData();
        }
        
        if (!trainFile) {
            throw new Error('Please upload train.csv file');
        }
        
        rawTrainData = await parseCSV(trainFile);
        
        if (testFile) {
            rawTestData = await parseCSV(testFile);
        } else {
            // For demo purposes, split train data if no test file
            const testSize = Math.floor(rawTrainData.length * 0.2);
            rawTestData = rawTrainData.slice(-testSize);
            rawTrainData = rawTrainData.slice(0, -testSize);
        }
        
        setStatus('loadStatus', `Loaded ${rawTrainData.length} training samples and ${rawTestData ? rawTestData.length : 0} test samples`, 'success');
        return true;
    } catch (error) {
        setStatus('loadStatus', `Failed to load data: ${error.message}`, 'error');
        return false;
    }
}

// ============================================================================
// DATA INSPECTION
// ============================================================================
function inspectData() {
    if (!rawTrainData || rawTrainData.length === 0) {
        setStatus('loadStatus', 'No data loaded. Please load data first.', 'error');
        return;
    }
    
    const inspectionContent = document.getElementById('inspectionContent');
    
    // Calculate basic statistics
    const sample = rawTrainData[0];
    const columns = Object.keys(sample);
    const totalRows = rawTrainData.length;
    
    // Calculate missing values
    let missingTable = '<h3>Missing Values Analysis</h3><table>';
    missingTable += '<tr><th>Column</th><th>Missing</th><th>Percentage</th></tr>';
    
    columns.forEach(col => {
        const missingCount = rawTrainData.filter(row => 
            row[col] === '' || row[col] === null || row[col] === undefined || (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        const percentage = ((missingCount / totalRows) * 100).toFixed(1);
        missingTable += `<tr>
            <td>${col}</td>
            <td>${missingCount}</td>
            <td>${percentage}%</td>
        </tr>`;
    });
    missingTable += '</table>';
    
    // Create data preview
    let previewTable = '<h3>Data Preview (First 5 Rows)</h3><div class="data-preview"><table>';
    previewTable += '<tr>' + columns.map(col => `<th>${col}</th>`).join('') + '</tr>';
    
    rawTrainData.slice(0, 5).forEach(row => {
        previewTable += '<tr>' + columns.map(col => {
            const val = row[col];
            return `<td>${val !== null && val !== undefined ? (typeof val === 'number' ? val.toFixed(2) : val) : 'N/A'}</td>`;
        }).join('') + '</tr>';
    });
    previewTable += '</table></div>';
    
    inspectionContent.innerHTML = `
        <div class="metric-box">
            <div class="metric-label">Dataset Shape:</div>
            <div class="metric-value">${totalRows} rows × ${columns.length} columns</div>
        </div>
        ${previewTable}
        ${missingTable}
    `;
    
    // Create visualizations if tfjs-vis is available
    createVisualizations();
    
    // Enable train button
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('evaluateBtn').disabled = false;
    document.getElementById('predictBtn').disabled = false;
}

function createVisualizations() {
    const chartsContainer = document.getElementById('chartsContainer');
    chartsContainer.innerHTML = '<h3>Data Distribution</h3><p>Visualizations will appear here during training.</p>';
    
    // Calculate survival statistics
    const survivalBySex = {};
    const survivalByClass = {};
    
    rawTrainData.forEach(row => {
        // Survival by Sex
        const sex = row.Sex || 'Unknown';
        if (!survivalBySex[sex]) {
            survivalBySex[sex] = { survived: 0, total: 0 };
        }
        survivalBySex[sex].total++;
        if (row.Survived === 1) survivalBySex[sex].survived++;
        
        // Survival by Class
        const pclass = row.Pclass || 'Unknown';
        if (!survivalByClass[pclass]) {
            survivalByClass[pclass] = { survived: 0, total: 0 };
        }
        survivalByClass[pclass].total++;
        if (row.Survived === 1) survivalByClass[pclass].survived++;
    });
    
    // Create data for tfjs-vis
    const surface1 = { name: 'Survival Rate by Sex', tab: 'Data Inspection' };
    const data1 = {
        values: Object.entries(survivalBySex).map(([sex, stats]) => ({
            x: sex,
            y: (stats.survived / stats.total) * 100
        }))
    };
    
    const surface2 = { name: 'Survival Rate by Passenger Class', tab: 'Data Inspection' };
    const data2 = {
        values: Object.entries(survivalByClass).map(([pclass, stats]) => ({
            x: `Class ${pclass}`,
            y: (stats.survived / stats.total) * 100
        }))
    };
    
    // Render charts
    if (typeof tfvis !== 'undefined') {
        tfvis.render.barchart(surface1, data1, {
            xLabel: 'Sex',
            yLabel: 'Survival Rate (%)',
            width: 400,
            height: 300
        });
        
        tfvis.render.barchart(surface2, data2, {
            xLabel: 'Passenger Class',
            yLabel: 'Survival Rate (%)',
            width: 400,
            height: 300
        });
    }
}

// ============================================================================
// PREPROCESSING PIPELINE
// ============================================================================
function preprocessData() {
    if (!rawTrainData) {
        throw new Error('No data loaded. Please load data first.');
    }
    
    // Combine train and test for consistent preprocessing
    const allData = [...rawTrainData];
    let testData = rawTestData ? [...rawTestData] : [];
    
    // Calculate statistics from training data only
    featureStats = calculateFeatureStatistics(rawTrainData);
    
    // Preprocess training data
    processedTrainData = {
        features: [],
        labels: [],
        passengerIds: []
    };
    
    rawTrainData.forEach(row => {
        const processed = preprocessRow(row, true);
        processedTrainData.features.push(processed.features);
        processedTrainData.labels.push(row[SCHEMA.TARGET] || 0);
        processedTrainData.passengerIds.push(row[SCHEMA.ID]);
    });
    
    // Preprocess test data if available
    processedTestData = {
        features: [],
        passengerIds: []
    };
    
    if (testData.length > 0) {
        testData.forEach(row => {
            const processed = preprocessRow(row, false);
            processedTestData.features.push(processed.features);
            processedTestData.passengerIds.push(row[SCHEMA.ID]);
        });
    }
    
    // Convert to tensors
    const trainFeatures = tf.tensor2d(processedTrainData.features);
    const trainLabels = tf.tensor1d(processedTrainData.labels).reshape([-1, 1]);
    
    // Store feature names for interpretation
    featureNames = processedTrainData.featureNames;
    
    // Update UI
    document.getElementById('trainSamples').textContent = processedTrainData.features.length;
    document.getElementById('valSamples').textContent = Math.floor(processedTrainData.features.length * 0.2);
    
    const featureList = document.getElementById('featureList');
    featureList.innerHTML = `
        <h3>Processed Features (${featureNames.length})</h3>
        <p>${featureNames.join(', ')}</p>
        <div class="metric-box">
            <div class="metric-label">Feature Tensor Shape:</div>
            <div class="metric-value">[${processedTrainData.features.length}, ${featureNames.length}]</div>
        </div>
    `;
    
    setStatus('preprocessStatus', `Preprocessed ${processedTrainData.features.length} samples with ${featureNames.length} features`, 'success');
    
    return { trainFeatures, trainLabels };
}

function calculateFeatureStatistics(data) {
    const stats = {};
    
    // For numerical features
    SCHEMA.NUMERICAL_FEATURES.forEach(feature => {
        const values = data.map(row => row[feature]).filter(val => val != null && !isNaN(val));
        if (values.length > 0) {
            stats[feature] = {
                mean: values.reduce((a, b) => a + b, 0) / values.length,
                std: Math.sqrt(values.reduce((sq, val) => sq + Math.pow(val - values.reduce((a, b) => a + b, 0) / values.length, 2), 0) / values.length),
                median: values.sort((a, b) => a - b)[Math.floor(values.length / 2)]
            };
        }
    });
    
    // For categorical features - calculate mode
    SCHEMA.CATEGORICAL_FEATURES.forEach(feature => {
        const valueCounts = {};
        data.forEach(row => {
            const val = row[feature];
            if (val != null && val !== '') {
                valueCounts[val] = (valueCounts[val] || 0) + 1;
            }
        });
        
        let mode = null;
        let maxCount = 0;
        Object.entries(valueCounts).forEach(([val, count]) => {
            if (count > maxCount) {
                mode = val;
                maxCount = count;
            }
        });
        
        stats[feature] = { mode };
    });
    
    return stats;
}

function preprocessRow(row, isTraining) {
    const features = [];
    const featureNames = [];
    
    // Process numerical features
    SCHEMA.NUMERICAL_FEATURES.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '' || isNaN(value)) {
            value = featureStats[feature] ? featureStats[feature].median : 0;
        }
        
        // Standardize
        if (featureStats[feature] && featureStats[feature].std > 0) {
            value = (value - featureStats[feature].mean) / featureStats[feature].std;
        }
        
        features.push(value);
        featureNames.push(feature);
    });
    
    // Process categorical features (one-hot encoding)
    SCHEMA.CATEGORICAL_FEATURES.forEach(feature => {
        let value = row[feature];
        
        // Handle missing values
        if (value === null || value === undefined || value === '') {
            value = featureStats[feature] ? featureStats[feature].mode : 'Unknown';
        }
        
        // Create unique categories based on training data statistics
        const categories = {};
        if (featureStats[feature] && featureStats[feature].categories) {
            featureStats[feature].categories.forEach((cat, idx) => {
                categories[cat] = idx;
            });
        } else {
            // For simplicity, we'll use common categories
            if (feature === 'Sex') {
                categories['male'] = 0;
                categories['female'] = 1;
            } else if (feature === 'Pclass') {
                categories[1] = 0;
                categories[2] = 1;
                categories[3] = 2;
            } else if (feature === 'Embarked') {
                categories['C'] = 0;
                categories['Q'] = 1;
                categories['S'] = 2;
            }
        }
        
        // One-hot encode
        const numCategories = Object.keys(categories).length;
        const oneHot = new Array(numCategories).fill(0);
        
        if (categories[value] !== undefined) {
            oneHot[categories[value]] = 1;
        } else if (numCategories > 0) {
            oneHot[0] = 1; // Default to first category
        }
        
        features.push(...oneHot);
        featureNames.push(...oneHot.map((_, idx) => `${feature}_${Object.keys(categories)[idx] || idx}`));
    });
    
    // Add derived features if enabled
    if (document.getElementById('addFamilyFeatures').checked) {
        Object.entries(SCHEMA.DERIVED_FEATURES).forEach(([name, func]) => {
            const value = func(row);
           
