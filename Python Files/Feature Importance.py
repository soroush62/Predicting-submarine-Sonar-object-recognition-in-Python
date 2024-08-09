import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the dashboard
dashboard = lc.Dashboard(
    columns=3,
    rows=2,
    theme=lc.Themes.Dark
)

def add_feature_importance_to_dashboard(dashboard, model_name, importances, column_index, row_index):
    """
    Add a feature importance bar chart to the dashboard.
    """
    importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    chart = dashboard.BarChart(
        column_index=column_index,
        row_index=row_index,
        row_span=1,
        column_span=1
    )
    chart.set_title(f'{model_name} Feature Importances')
    
    # Prepare data for the BarChart, ensuring all values are native Python data types
    bar_data = [{'category': str(row['Feature']), 'value': float(row['Importance'])} for _, row in importance_df.iterrows()]
    chart.set_data(bar_data)

for i, (model_name, model) in enumerate(models.items()):
    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(selected_features))

    # Add the feature importance chart to the dashboard
    add_feature_importance_to_dashboard(
        dashboard=dashboard,
        model_name=model_name,
        importances=importances,
        column_index=i % 3,
        row_index=i // 3
    )

# Ensemble Methods (excluding models without feature_importances_)
estimators = [
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')

# Create a pipeline with the preprocessor and the ensemble model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# Train the ensemble model
pipeline.fit(X_train, y_train)

# Extract feature importances
ensemble_importances = np.mean([
    pipeline.named_steps['classifier'].estimators_[i].feature_importances_
    for i in range(len(pipeline.named_steps['classifier'].estimators_))
], axis=0)

# Add the ensemble feature importance chart to the dashboard
add_feature_importance_to_dashboard(
    dashboard=dashboard,
    model_name='Ensemble Methods',
    importances=ensemble_importances,
    column_index=2,
    row_index=1
)

# Open the dashboard in the default web browser
dashboard.open()
