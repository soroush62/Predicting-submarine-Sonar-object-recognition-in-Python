import numpy as np
import pandas as pd
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
data = pd.read_csv(file_path, header=None)

X = data.drop(columns=[60])
y = data[60]

mean_values_rocks = X[y == 'R'].mean().tolist()
mean_values_mines = X[y == 'M'].mean().tolist()

feature_labels = [str(i) for i in range(60)]

chart = lc.SpiderChart(
    theme=lc.Themes.White,
    title='Mean Feature Values for Rocks and Mines'
)

for label in feature_labels:
    chart.add_axis(label)

series_rocks = chart.add_series()
series_rocks.set_name('Rocks')
series_rocks.add_points([
    {'axis': label, 'value': value} for label, value in zip(feature_labels, mean_values_rocks)
])

series_mines = chart.add_series()
series_mines.set_name('Mines')
series_mines.add_points([
    {'axis': label, 'value': value} for label, value in zip(feature_labels, mean_values_mines)
])

chart.open()
