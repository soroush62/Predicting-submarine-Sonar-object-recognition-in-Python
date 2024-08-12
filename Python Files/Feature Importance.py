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

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
data = pd.read_csv(file_path, header=None)

X = data.drop(columns=[60])
y = data[60].apply(lambda x: 1 if x == 'M' else 0)  # Convert target to binary values

selected_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features)
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    
    bar_data = [{'category': str(row['Feature']), 'value': float(row['Importance'])} for _, row in importance_df.iterrows()]
    chart.set_data(bar_data)

for i, (model_name, model) in enumerate(models.items()):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(selected_features))

    add_feature_importance_to_dashboard(
        dashboard=dashboard,
        model_name=model_name,
        importances=importances,
        column_index=i % 3,
        row_index=i // 3
    )

estimators = [
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

pipeline.fit(X_train, y_train)

ensemble_importances = np.mean([
    pipeline.named_steps['classifier'].estimators_[i].feature_importances_
    for i in range(len(pipeline.named_steps['classifier'].estimators_))
], axis=0)

add_feature_importance_to_dashboard(
    dashboard=dashboard,
    model_name='Ensemble Methods',
    importances=ensemble_importances,
    column_index=2,
    row_index=1
)

dashboard.open()
