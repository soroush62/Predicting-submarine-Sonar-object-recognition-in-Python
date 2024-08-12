import pandas as pd
import numpy as np
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
sonar_data = pd.read_csv(file_path, header=None)

sonar_features = sonar_data.drop(columns=[60])

corr_matrix = sonar_features.corr()

heatmap_data = corr_matrix.values.tolist()

chart = lc.ChartXY(
    theme=lc.Themes.White,
    title='Correlation Heatmap of Sonar Features'
)
series = chart.add_heatmap_grid_series(
    columns=len(heatmap_data),
    rows=len(heatmap_data[0])
)

series.hide_wireframe()
series.set_intensity_interpolation(False)
series.invalidate_intensity_values(heatmap_data)

series.set_palette_colors(
    steps=[
        {"value": -1.0, "color": lc.Color(255,255, 255)},  # Blue for negative correlation
        {"value": 0.0, "color": lc.Color(0, 0, 255)},  # White for no correlation
        {"value": 1.0, "color": lc.Color(255, 0, 0)}  # Red for positive correlation
    ],
    look_up_property='value',
    percentage_values=True
)

x_axis = chart.get_default_x_axis()
x_axis.set_title('Feature Index')
x_axis.set_interval(0, sonar_features.shape[1])

y_axis = chart.get_default_y_axis()
y_axis.set_title('Feature Index')
y_axis.set_interval(0, sonar_features.shape[1])

chart.open()
