# -Task3_-Human-Heart-Disease

Learning
📌 Overview
This project focuses on building a machine learning pipeline to predict the presence of heart disease based on medical attributes. The workflow covers data preprocessing, feature transformation, model training, and evaluation using Python's scikit-learn library.

📂 Files in Repository
Task3(HeartDisease).ipynb – Jupyter Notebook containing the full workflow for heart disease prediction.

🛠️ Requirements
Install the required dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
📋 Steps Performed
1. Data Loading & Exploration
Loaded heart disease dataset into Pandas DataFrame.

Checked shape, missing values, and data types.

Generated basic statistics and visualized correlations using seaborn heatmaps.

2. Feature & Target Separation
Defined X as input features and y as target (target_col).

3. Data Preprocessing Pipeline
Identified numeric and categorical columns.

Built a preprocessing pipeline using:

Numeric Transformer:

SimpleImputer(strategy='median') for missing values.

StandardScaler() for normalization.

Categorical Transformer:

SimpleImputer(strategy='most_frequent') for missing values.

OneHotEncoder(handle_unknown='ignore', sparse_output=False) for encoding.

Combined transformers using ColumnTransformer.

4. Train-Test Split
Split dataset into training and testing sets using train_test_split (80-20 split).

5. Model Training
Trained and evaluated multiple models:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Used accuracy score, confusion matrix, and classification report for evaluation.

6. Results
Compared models to find the best-performing algorithm.

📊 Example Visualizations
Heatmaps to show feature correlations.

Confusion Matrices to evaluate classification results.

▶️ Running the Notebook
Open Task3(HeartDisease).ipynb in Jupyter Notebook, JupyterLab, or Google Colab.

Ensure dataset file is in the correct location.

Run cells in sequence to reproduce results.
