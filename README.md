# kuznetsovv_nndl

Output two separate code files: index.html (HTML structure, UI, basic CSS) and app.js (all JavaScript logic).

The app is a shallow binary classifier trained on the Kaggle Titanic dataset, built with TensorFlow.js, running entirely in the browser (no server) and ready for GitHub Pages deployment.

Use these CDNs only:
- TensorFlow.js: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest
- tfjs-vis: https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest

Link app.js from index.html.

Layout (index.html):
Create clear UI sections with headers: Data Load, Data Inspection, Preprocessing, Model, Training, Evaluation & Metrics, Prediction, Export, Deployment Notes.
Include file inputs for train.csv and test.csv, buttons (Inspect Data, Train, Evaluate, Predict), and placeholder containers for data preview table, evaluation table, ROC curve, confusion matrix, and feature-importance visualization. Use basic responsive CSS (flex/grid, mobile-friendly).

Data Schema (app.js):
Target: Survived (0/1).
Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
Identifier (exclude from training): PassengerId.
Add comments explaining where to swap schema for another dataset.

CSV Loading & Inspection:
Fix the comma-escape problem when loading CSV files. Passenger names may contain commas inside quotes (e.g. "Braund, Mr. Owen Harris"), so do NOT parse CSV using a naive split(','). Use a robust CSV parsing approach that correctly handles quoted fields, commas inside quotes, and missing values (regex-based or PapaParse-style logic). If parsing fails, show a clear alert indicating the problematic row and reason.
After loading, display the first 10 rows as an HTML table, show dataset shape (rows × columns), calculate and display missing-value percentage per column, and visualize survival rates by Sex and by Pclass using tfjs-vis bar charts.

Preprocessing:
Impute Age with the median and Embarked with the mode. Add feature-engineering toggles for FamilySize = SibSp + Parch + 1 and IsAlone = (FamilySize === 1). One-hot encode Sex, Pclass, and Embarked. Standardize Age and Fare. Log the final feature list and tensor shapes (X and y).

Model:
Build a tf.sequential model with Dense(16, activation='relu') followed by Dense(1, activation='sigmoid'). Compile with optimizer 'adam', loss 'binaryCrossentropy', and metric 'accuracy'. Print the model summary to the console.

Training:
Perform an 80/20 stratified train/validation split. Train for 50 epochs with batch size 32. Use early stopping on val_loss with patience = 5. Visualize live training loss and accuracy using tfjs-vis fitCallbacks.

Evaluation & Metrics:
Fix the issue where evaluation results are computed but not shown. Render an evaluation table in the UI containing Accuracy, Precision, Recall, F1-score, and ROC-AUC. Ensure the table appears only after evaluation and updates dynamically when the threshold changes.
Compute validation probabilities, generate and plot an ROC curve using tfjs-vis, and add a threshold slider (0–1) that dynamically updates the confusion matrix and Precision/Recall/F1 metrics.

Sigmoid Gate & Feature Importance:
Add an interpretability module using a sigmoid gating mechanism (or equivalent post-hoc logic) that learns a weight per input feature and outputs values in the range [0,1]. Treat these values as relative feature importance scores. Visualize feature importance with a tfjs-vis bar chart labeled “Relative Feature Importance (Sigmoid Gate)”. Clearly document in comments that this is a lightweight interpretability heuristic and not SHAP.

Prediction & Export:
Run inference on test.csv to produce survival probabilities, apply the selected threshold to generate binary predictions, and allow downloading of submission.csv (PassengerId, Survived) and probabilities.csv (PassengerId, Probability). Enable exporting the trained model via model.save('downloads://titanic-tfjs').

Self-Review & Code Summary:
At the end of app.js, include a commented section titled “LLM SELF-REVIEW SUMMARY” that explains: (1) how CSV parsing avoids comma-in-quote bugs, (2) how preprocessing transforms the raw Titanic data, (3) how the model is trained and evaluated, (4) how ROC and thresholding work, and (5) how the sigmoid gate provides feature importance. This summary should clearly describe the full logic flow of the application.

Deployment Notes (index.html):
Include visible text instructions explaining how to create a public GitHub repository, commit index.html and app.js, enable GitHub Pages (main/root), and open the generated URL.

General Requirements:
All code must include clear English comments. Buttons must be fully interactive. Handle errors gracefully (missing files, invalid CSV, training before loading data). Ensure the code is readable, modular, reusable, and easy to adapt to other datasets by swapping the schema.
