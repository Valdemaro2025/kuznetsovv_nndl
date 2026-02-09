        // Normalize derived features (simple min-max scaling)
        const normalizedValue = value / 10; // Simple scaling for demo
        features.push(normalizedValue);
        featureNames.push(name);
    });
    
    return { features, featureNames };
}

// ============================================================================
// MODEL ARCHITECTURE
// ============================================================================
function createModel(inputShape) {
    const model = tf.sequential();
    
    // Feature importance gate (optional)
    const useFeatureGate = document.getElementById('useFeatureGate').checked;
    
    if (useFeatureGate) {
        // Add trainable sigmoid gate for feature importance
        model.add(tf.layers.dense({
            units: inputShape,
            activation: 'sigmoid',
            useBias: false,
            kernelInitializer: 'ones',
            trainable: true,
            name: 'feature_gate'
        }));
    }
    
    // Main model architecture
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape],
        kernelInitializer: 'heNormal'
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Store reference to gate layer for later analysis
    if (useFeatureGate) {
        featureGateLayer = model.layers[0];
    }
    
    // Display model summary
    const modelSummary = document.getElementById('modelSummary');
    let summaryText = '<h3>Model Architecture</h3>';
    summaryText += `<p>Input: ${inputShape} features</p>`;
    summaryText += '<p>Layers:</p><ul>';
    
    model.layers.forEach((layer, idx) => {
        summaryText += `<li>${layer.name}: ${layer.outputShape}</li>`;
    });
    
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams().toLocaleString()}</p>`;
    modelSummary.innerHTML = summaryText;
    
    return model;
}

// ============================================================================
// TRAINING
// ============================================================================
async function trainModel() {
    try {
        // Preprocess data first
        const { trainFeatures, trainLabels } = preprocessData();
        
        // Create validation split (80/20)
        const splitIndex = Math.floor(trainFeatures.shape[0] * 0.8);
        const xTrain = trainFeatures.slice([0, 0], [splitIndex, -1]);
        const yTrain = trainLabels.slice([0, 0], [splitIndex, -1]);
        const xVal = trainFeatures.slice([splitIndex, 0], [-1, -1]);
        const yVal = trainLabels.slice([splitIndex, 0], [-1, -1]);
        
        validationData = { xVal, yVal };
        
        // Create model
        model = createModel(trainFeatures.shape[1]);
        
        // Setup training callbacks
        const surface = { name: 'Training Metrics', tab: 'Training' };
        const callbacks = tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
            callbacks: ['onEpochEnd'],
            height: 300
        });
        
        // Add early stopping
        callbacks.onEpochEnd = async (epoch, logs) => {
            if (callbacks.onEpochEndOriginal) {
                callbacks.onEpochEndOriginal(epoch, logs);
            }
            
            // Early stopping logic
            if (epoch >= 5) {
                const currentValLoss = logs.val_loss;
                if (currentValLoss > (trainingHistory?.val_losses?.[epoch-5] || Infinity)) {
                    console.log(`Early stopping triggered at epoch ${epoch}`);
                    model.stopTraining = true;
                }
            }
            
            // Store history
            if (!trainingHistory) {
                trainingHistory = {
                    losses: [],
                    val_losses: [],
                    accs: [],
                    val_accs: []
                };
            }
            trainingHistory.losses.push(logs.loss);
            trainingHistory.val_losses.push(logs.val_loss);
            trainingHistory.accs.push(logs.acc);
            trainingHistory.val_accs.push(logs.val_acc);
            
            // Update UI
            setStatus('trainingStatus', `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Acc: ${(logs.acc * 100).toFixed(1)}%`, 'success');
        };
        
        callbacks.onEpochEndOriginal = callbacks.onEpochEnd;
        
        // Train model
        setStatus('trainingStatus', 'Training started... This may take a moment.', 'info');
        
        const history = await model.fit(xTrain, yTrain, {
            epochs: 50,
            batchSize: 32,
            validationData: [xVal, yVal],
            callbacks: callbacks,
            verbose: 0
        });
        
        // Clean up tensors
        xTrain.dispose();
        yTrain.dispose();
        trainFeatures.dispose();
        trainLabels.dispose();
        
        setStatus('trainingStatus', 'Training completed successfully!', 'success');
        
        // Enable export buttons
        document.getElementById('exportModelBtn').disabled = false;
        
        return history;
    } catch (error) {
        setStatus('trainingStatus', `Training failed: ${error.message}`, 'error');
        throw error;
    }
}

// ============================================================================
// EVALUATION & METRICS
// ============================================================================
function evaluateModel() {
    if (!model || !validationData) {
        setStatus('loadStatus', 'Please train the model first', 'error');
        return;
    }
    
    try {
        const { xVal, yVal } = validationData;
        
        // Get predictions
        const predictions = model.predict(xVal);
        const predValues = predictions.dataSync();
        const trueValues = yVal.dataSync();
        
        // Calculate ROC curve
        const rocData = calculateROCCurve(predValues, trueValues);
        
        // Update AUC
        document.getElementById('aucScore').textContent = rocData.auc.toFixed(3);
        
        // Update threshold slider
        updateMetrics(0.5);
        
        // Plot ROC curve
        plotROCCurve(rocData);
        
        // Extract feature importance if gate exists
        if (featureGateLayer) {
            displayFeatureImportance();
        }
        
        setStatus('loadStatus', 'Evaluation completed. Adjust threshold slider to see metrics.', 'success');
        
        // Enable prediction button
        document.getElementById('predictBtn').disabled = false;
        
        // Clean up
        predictions.dispose();
    } catch (error) {
        setStatus('loadStatus', `Evaluation failed: ${error.message}`, 'error');
    }
}

function calculateROCCurve(predictions, trueLabels) {
    // Generate thresholds
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    
    const tpr = []; // True Positive Rate (Recall)
    const fpr = []; // False Positive Rate
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        predictions.forEach((pred, idx) => {
            const predictedClass = pred >= threshold ? 1 : 0;
            const trueClass = trueLabels[idx];
            
            if (trueClass === 1 && predictedClass === 1) tp++;
            if (trueClass === 0 && predictedClass === 1) fp++;
            if (trueClass === 0 && predictedClass === 0) tn++;
            if (trueClass === 1 && predictedClass === 0) fn++;
        });
        
        tpr.push(tp / (tp + fn) || 0);
        fpr.push(fp / (fp + tn) || 0);
    });
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < thresholds.length; i++) {
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
    }
    
    return { tpr, fpr, thresholds, auc };
}

function plotROCCurve(rocData) {
    const rocContainer = document.getElementById('rocContainer');
    rocContainer.innerHTML = '<h3>ROC Curve</h3>';
    
    if (typeof tfvis !== 'undefined') {
        const surface = { name: 'ROC Curve', tab: 'Evaluation' };
        
        const rocValues = rocData.thresholds.map((t, i) => ({
            x: rocData.fpr[i],
            y: rocData.tpr[i],
            threshold: t
        }));
        
        tfvis.render.scatterplot(surface, { values: rocValues }, {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 400,
            height: 400,
            color: '#2196f3'
        });
        
        // Add diagonal line reference
        const referenceLine = {
            values: [{x: 0, y: 0}, {x: 1, y: 1}]
        };
        
        tfvis.render.linechart(surface, referenceLine, {
            series: ['Random'],
            width: 400,
            height: 400
        });
    } else {
        rocContainer.innerHTML += '<p>ROC curve visualization requires tfjs-vis</p>';
    }
}

function updateMetrics(threshold) {
    if (!model || !validationData) return;
    
    const { xVal, yVal } = validationData;
    const predictions = model.predict(xVal);
    const predValues = predictions.dataSync();
    const trueValues = yVal.dataSync();
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    predValues.forEach((pred, idx) => {
        const predictedClass = pred >= threshold ? 1 : 0;
        const trueClass = trueValues[idx];
        
        if (trueClass === 1 && predictedClass === 1) tp++;
        if (trueClass === 0 && predictedClass === 1) fp++;
        if (trueClass === 0 && predictedClass === 0) tn++;
        if (trueClass === 1 && predictedClass === 0) fn++;
    });
    
    // Update confusion matrix display
    const confusionDiv = document.getElementById('confusionMatrix');
    confusionDiv.innerHTML = `
        <div></div>
        <div class="confusion-cell" style="background: rgba(33, 150, 243, 0.2);">Predicted 0</div>
        <div class="confusion-cell" style="background: rgba(33, 150, 243, 0.2);">Predicted 1</div>
        <div class="confusion-cell" style="background: rgba(33, 150, 243, 0.2);">Actual 0</div>
        <div class="confusion-cell true-negative">${tn}</div>
        <div class="confusion-cell false-positive">${fp}</div>
        <div class="confusion-cell" style="background: rgba(33, 150, 243, 0.2);">Actual 1</div>
        <div class="confusion-cell false-negative">${fn}</div>
        <div class="confusion-cell true-positive">${tp}</div>
    `;
    
    // Calculate metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Update metric displays
    document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
    document.getElementById('precision').textContent = precision.toFixed(3);
    document.getElementById('recall').textContent = recall.toFixed(3);
    document.getElementById('f1Score').textContent = f1.toFixed(3);
    
    // Clean up
    predictions.dispose();
}

// ============================================================================
// FEATURE IMPORTANCE
// ============================================================================
function displayFeatureImportance() {
    if (!featureGateLayer || !featureNames) return;
    
    // Get gate weights
    const weights = featureGateLayer.getWeights()[0];
    const importanceValues = Array.from(weights.dataSync());
    
    // Normalize importance scores
    const maxImportance = Math.max(...importanceValues);
    const normalized = importanceValues.map(val => val / maxImportance);
    
    // Display feature importance
    const featureImportanceDiv = document.getElementById('featureImportance');
    let html = '<h3>Learned Feature Importance</h3>';
    
    // Create importance bars
    normalized.forEach((importance, idx) => {
        if (idx < featureNames.length) {
            const featureName = featureNames[idx];
            const width = (importance * 100).toFixed(1);
            html += `
                <div style="margin: 8px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${featureName}</span>
                        <span>${importance.toFixed(3)}</span>
                    </div>
                    <div class="importance-bar" style="width: ${width}%;"></div>
                </div>
            `;
        }
    });
    
    featureImportanceDiv.innerHTML = html;
}

// ============================================================================
// PREDICTION & EXPORT
// ============================================================================
async function generatePredictions() {
    if (!model || !processedTestData) {
        setStatus('predictionStatus', 'Please load test data and train model first', 'error');
        return;
    }
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(processedTestData.features);
        
        // Generate predictions
        const predictions = model.predict(testFeatures);
        const probabilities = Array.from(predictions.dataSync());
        
        // Store for export
        testPredictions = probabilities;
        testPassengerIds = processedTestData.passengerIds;
        
        // Apply threshold
        const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
        const predictedClasses = probabilities.map(p => p >= threshold ? 1 : 0);
        
        // Show sample predictions
        let sampleHtml = '<h3>Sample Predictions</h3><table>';
        sampleHtml += '<tr><th>PassengerId</th><th>Probability</th><th>Predicted</th></tr>';
        
        const sampleCount = Math.min(10, probabilities.length);
        for (let i = 0; i < sampleCount; i++) {
            sampleHtml += `<tr>
                <td>${testPassengerIds[i]}</td>
                <td>${probabilities[i].toFixed(4)}</td>
                <td>${predictedClasses[i]}</td>
            </tr>`;
        }
        
        sampleHtml += '</table>';
        
        // Update status
        setStatus('predictionStatus', `Generated ${probabilities.length} predictions`, 'success');
        
        const statusDiv = document.getElementById('predictionStatus');
        statusDiv.innerHTML = `<i class="fas fa-check-circle"></i> Generated ${probabilities.length} predictions` + sampleHtml;
        
        // Enable export buttons
        document.getElementById('exportSubmissionBtn').disabled = false;
        document.getElementById('exportProbabilitiesBtn').disabled = false;
        
        // Clean up
        testFeatures.dispose();
        predictions.dispose();
    } catch (error) {
        setStatus('predictionStatus', `Prediction failed: ${error.message}`, 'error');
    }
}

function exportSubmissionCSV() {
    if (!testPredictions || !testPassengerIds) {
        alert('Please generate predictions first');
        return;
    }
    
    const threshold = parseFloat(document.getElementById('thresholdValue').textContent);
    let csvContent = 'PassengerId,Survived\n';
    
    testPredictions.forEach((prob, idx) => {
        const passengerId = testPassengerIds[idx];
        const survived = prob >= threshold ? 1 : 0;
        csvContent += `${passengerId},${survived}\n`;
    });
    
    downloadCSV(csvContent, 'submission.csv');
}

function exportProbabilitiesCSV() {
    if (!testPredictions || !testPassengerIds) {
        alert('Please generate predictions first');
        return;
    }
    
    let csvContent = 'PassengerId,Probability\n';
    
    testPredictions.forEach((prob, idx) => {
        const passengerId = testPassengerIds[idx];
        csvContent += `${passengerId},${prob.toFixed(6)}\n`;
    });
    
    downloadCSV(csvContent, 'probabilities.csv');
}

async function exportModel() {
    if (!model) {
        alert('Please train a model first');
        return;
    }
    
    try {
        // Save model locally
        await model.save('downloads://titanic-model');
        setStatus('trainingStatus', 'Model downloaded successfully', 'success');
    } catch (error) {
        setStatus('trainingStatus', `Failed to export model: ${error.message}`, 'error');
    }
}

// ============================================================================
// EVENT LISTENERS SETUP
// ============================================================================
function setupEventListeners() {
    // Load data button
    document.getElementById('loadBtn').addEventListener('click', async () => {
        const success = await loadData();
        if (success) {
            inspectData();
        }
    });
    
    // Train model button
    document.getElementById('trainBtn').addEventListener('click', async () => {
        try {
            await trainModel();
        } catch (error) {
            console.error('Training error:', error);
        }
    });
    
    // Evaluate button
    document.getElementById('evaluateBtn').addEventListener('click', () => {
        evaluateModel();
    });
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', () => {
        generatePredictions();
    });
    
    // Threshold slider
    document.getElementById('thresholdSlider').addEventListener('input', (e) => {
        const threshold = parseFloat(e.target.value);
        updateMetrics(threshold);
    });
    
    // Export buttons
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmissionCSV);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilitiesCSV);
    document.getElementById('exportModelBtn').addEventListener('click', exportModel);
    
    // File input changes trigger load
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
    setStatus('loadStatus', 'Ready to load Titanic dataset. Upload CSV files or use demo data.', 'info');
});

// ============================================================================
// CODE SUMMARY
// ============================================================================
/*
DATA FLOW:
1. CSV files loaded via file input or demo data fetched from GitHub
2. Raw data parsed and stored in memory
3. Data inspection shows statistics and visualizations
4. Preprocessing: missing value imputation, standardization, one-hot encoding
5. Data split into training (80%) and validation (20%) sets
6. Model training with early stopping and live metrics visualization
7. Evaluation: ROC curve, AUC, confusion matrix, precision/recall/F1
8. Prediction on test set with adjustable threshold
9. Export: submission CSV, probabilities CSV, trained model

PREPROCESSING PIPELINE:
- Numerical features: median imputation + standardization
- Categorical features: mode imputation + one-hot encoding
- Derived features: FamilySize, IsAlone (toggleable)
- Schema swap points: SCHEMA object at top of file

MODEL ARCHITECTURE:
- Optional learnable feature importance gate (sigmoid activation)
- Dense(16, relu) hidden layer
- Dense(1, sigmoid) output layer
- Binary crossentropy loss, Adam optimizer

TRAINING LOGIC:
- 50 epochs maximum with early stopping (patience=5)
- Batch size: 32
- Live loss/accuracy plots via tfjs-vis
- Validation split for monitoring overfitting

EVALUATION LOGIC:
- ROC curve and AUC calculation
- Dynamic confusion matrix based on threshold
- Precision, recall, F1 score computation
- Feature importance visualization from gate layer

EXPORT LOGIC:
- Kaggle submission format CSV
- Raw probabilities CSV
- Model download via TensorFlow.js save() API

REUSABILITY:
- Modify SCHEMA object for different datasets
- Adjust preprocessing in preprocessRow() function
- Update visualization functions as needed
*/
