import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Read the license key from a file
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the sonar dataset
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

def add_roc_curve_to_dashboard(dashboard, model_name, model, column_index, row_index):    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    thresholds = np.nan_to_num(thresholds, nan=0.0)
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    
    chart = dashboard.ChartXY(column_index=column_index, row_index=row_index, column_span=1, row_span=1)
    chart.set_title(f'{model_name} ROC Curve with Thresholds (AUC = {roc_auc:.2f})')
    
    roc_series = chart.add_line_series()
    roc_series.add(fpr.tolist(), tpr.tolist()).set_name('ROC Curve')
    
    diagonal_series = chart.add_line_series()
    diagonal_series.add([0, 1], [0, 1])
    diagonal_series.set_name('Dashed Line')
    diagonal_series.set_dashed(pattern='Dashed')
    
    point_series = chart.add_point_series().set_name('Threshold Points')
    
    for j in range(len(thresholds)):
        color = lc.Color(
            int(255 * (1 - normalized_thresholds[j])),  # Red component decreases with threshold
            int(255 * normalized_thresholds[j]),        # Green component increases with threshold
            0                                           # Blue component stays constant
        )
        point_series.add(fpr[j], tpr[j]).set_point_color(color)
    
    chart.get_default_x_axis().set_title('False Positive Rate')
    chart.get_default_y_axis().set_title('True Positive Rate')    
    legend = chart.add_legend(horizontal=False)
    legend.add(roc_series)
    legend.add(point_series)

# Add ROC curves for individual models
for i, (model_name, model) in enumerate(models.items()):
    add_roc_curve_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

# Ensemble Methods
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

add_roc_curve_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

dashboard.open()
