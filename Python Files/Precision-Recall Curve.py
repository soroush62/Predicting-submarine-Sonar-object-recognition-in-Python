import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read the license key from a file
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the dataset
file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
data = pd.read_csv(file_path, header=None)

# Prepare the features and target
X = data.drop(columns=[60])
y = data[60].apply(lambda x: 1 if x == 'M' else 0)  # Convert target to binary values

# Preprocess the data (scale numerical features)
selected_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features)
    ])

# Define the models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the dashboard
dashboard = lc.Dashboard(rows=2, columns=3, theme=lc.Themes.Dark)

def add_pr_curve_to_dashboard(dashboard, model_name, model, column_index, row_index):    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    chart = dashboard.ChartXY(column_index=column_index, row_index=row_index, column_span=1, row_span=1)
    chart.set_title(f'{model_name} Precision-Recall Curve (AUC = {pr_auc:.2f})')
    
    pr_series = chart.add_line_series()
    pr_series.add(recall.tolist(), precision.tolist()).set_name('Precision-Recall Curve')
    
    baseline_series = chart.add_line_series()
    baseline_series.add([0, 1], [1, 0])
    baseline_series.set_name('Baseline')
    baseline_series.set_dashed(pattern='Dashed')
    
    chart.get_default_x_axis().set_title('Recall')
    chart.get_default_y_axis().set_title('Precision')    
    legend = chart.add_legend(horizontal=False)
    legend.add(pr_series)
    legend.add(baseline_series)

# Add Precision-Recall curves for individual models
for i, (model_name, model) in enumerate(models.items()):
    add_pr_curve_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

# Ensemble Methods
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

add_pr_curve_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

dashboard.open()
