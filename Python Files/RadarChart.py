import numpy as np
import pandas as pd
import lightningchart as lc

# Read the license key from a file
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the dataset
file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
data = pd.read_csv(file_path, header=None)

# Separate features and labels
X = data.drop(columns=[60])
y = data[60]

# Calculate mean values for each feature for Rocks and Mines
mean_values_rocks = X[y == 'R'].mean().tolist()
mean_values_mines = X[y == 'M'].mean().tolist()

# Define the feature labels
feature_labels = [str(i) for i in range(60)]

# Create a Spider Chart
chart = lc.SpiderChart(
    theme=lc.Themes.White,
    title='Mean Feature Values for Rocks and Mines'
)

# Add features as axes
for label in feature_labels:
    chart.add_axis(label)

# Add series for Rocks
series_rocks = chart.add_series()
series_rocks.set_name('Rocks')
series_rocks.add_points([
    {'axis': label, 'value': value} for label, value in zip(feature_labels, mean_values_rocks)
])

# Add series for Mines
series_mines = chart.add_series()
series_mines.set_name('Mines')
series_mines.add_points([
    {'axis': label, 'value': value} for label, value in zip(feature_labels, mean_values_mines)
])

# Open the chart in the default web browser
chart.open()
