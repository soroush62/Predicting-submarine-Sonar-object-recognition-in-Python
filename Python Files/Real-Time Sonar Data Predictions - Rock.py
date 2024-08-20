import numpy as np
import pandas as pd
import lightningchart as lc
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

lc.set_license('my-license-key')

file_path = 'sonar.csv'
data = pd.read_csv(file_path, header=None)

X = data.drop(columns=[60])
y = data[60].apply(lambda x: 1 if x == 'M' else 0)  # 1 for Mine, 0 for Rock

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

def generate_sonar_data():
    """
    Generate simulated sonar data with more variability
    to potentially trigger both "Rock" and "Mine" predictions.
    """
    base_signal = np.sin(np.linspace(0, 2 * np.pi, 60))  # Base sine wave
    noise = np.random.normal(0, 0.2, 60)  # Add noise to the signal
    variation = np.random.choice([0.5, 1, 1.5, 2], 1)[0]  # Random amplitude variation
    simulated_data = base_signal * variation + noise
    return np.round(simulated_data, 3)

dashboard = lc.Dashboard(rows=2, columns=1, theme=lc.Themes.Dark)

chart = dashboard.ChartXY(row_index=0, column_index=0)
series = chart.add_line_series(data_pattern='ProgressiveX')
chart.get_default_x_axis().set_scroll_strategy(strategy='progressive')
chart.get_default_x_axis().set_interval(start=-500, end=0, stop_axis_after=False)
chart.get_default_y_axis().set_title('Value')
chart.set_title('Real-Time Sonar Data Predictions')

bar_chart = dashboard.BarChart(row_index=1, column_index=0)
bar_chart.set_title('Cumulative Predictions')

bar_data = {'Rock': 0, 'Mine': 0}
bar_chart.set_data([{'category': key, 'value': value} for key, value in bar_data.items()])

dashboard.open(live=True)

def update_chart():
    global bar_data
    
    x_counter = 0
    while True:
        data = generate_sonar_data().reshape(1, -1)
        
        prediction = pipeline.predict(data)[0]
        label = 'Mine' if prediction == 1 else 'Rock'
        
        y_values = data.flatten().tolist()    # Generated sonar values
        
        for y_value in y_values:
            series.add(x_counter, y_value)
            x_counter += 1
        
        chart.set_title(f'Real-Time Sonar Data Predictions - {label}')
        
        bar_data[label] += 1
        bar_chart.set_data([{'category': key, 'value': value} for key, value in bar_data.items()])
        
        time.sleep(1.0)

update_chart()

dashboard.close()



