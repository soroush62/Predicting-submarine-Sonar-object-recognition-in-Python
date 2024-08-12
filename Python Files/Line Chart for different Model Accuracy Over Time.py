import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
sonar_data = pd.read_csv(file_path, header=None)

X = sonar_data.drop(columns=[60])
y = sonar_data[60].apply(lambda x: 1 if x == 'M' else 0)  # Convert target to binary values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=250, solver='liblinear'),
    'Support Vector Machine': SVC(max_iter=250),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

iterations = 5

dashboard = lc.Dashboard(
    rows=2, columns=3, theme=lc.Themes.Dark
)

# Iterate over models
for idx, (model_name, model) in enumerate(models.items()):
    train_accuracies = []
    test_accuracies = []
    
    for i in range(1, iterations + 1):
        if hasattr(model, 'max_iter'):
            model.set_params(max_iter=i * 50)
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

    chart = dashboard.ChartXY(
        row_index=idx // 3,
        column_index=idx % 3,
        title=model_name
    )

    series_train = chart.add_line_series()
    series_train.add(range(1, iterations + 1), train_accuracies)
    series_train.set_name('Training Accuracy')
    series_train.set_line_color(lc.Color('blue'))

    markers_train = chart.add_point_series()
    markers_train.add(range(1, iterations + 1), train_accuracies)
    markers_train.set_point_shape('circle')
    markers_train.set_point_size(8)
    markers_train.set_point_color(lc.Color('blue'))

    series_test = chart.add_line_series()
    series_test.add(range(1, iterations + 1), test_accuracies)
    series_test.set_name('Testing Accuracy')
    series_test.set_line_color(lc.Color('orange'))

    markers_test = chart.add_point_series()
    markers_test.add(range(1, iterations + 1), test_accuracies)
    markers_test.set_point_shape('triangle')
    markers_test.set_point_size(8)
    markers_test.set_point_color(lc.Color('orange'))

    chart.get_default_x_axis().set_title('Iteration')
    chart.get_default_y_axis().set_title('Accuracy').set_interval(0, 1)

    legend = chart.add_legend()
    legend.add(data=series_test)
    legend.add(data=series_train)

dashboard.open()
