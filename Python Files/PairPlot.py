




import numpy as np
import pandas as pd
import lightningchart as lc
from scipy.stats import gaussian_kde

# Read the license key from a file
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the sonar dataset
file_path = 'D:/wenprograming23/src/team6/Predicting-submarine-Sonar-object-recognition-in-Python/Dataset/sonar.csv'
sonar_data = pd.read_csv(file_path, header=None)

# Separate features and labels
X = sonar_data.drop(columns=[60])
y = sonar_data[60]

# Add the target variable to the dataset for visualization
sonar_data['Target'] = y

# Extract the relevant columns for the pairplot
columns = [0, 1, 2, 3]
target_col = 'Target'

# Create a Dashboard
dashboard = lc.Dashboard(
    rows=len(columns),
    columns=len(columns),
    theme=lc.Themes.Dark
)

def create_scatter_chart(dashboard, title, x_values, y_values, xlabel, ylabel, column_index, row_index):
    # Create the scatter plot chart
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    # Add the points to the scatter chart
    scatter_series = chart.add_point_series()
    scatter_series.add(x_values, y_values)

    # Set axis labels
    chart.get_default_x_axis().set_title(xlabel)
    chart.get_default_y_axis().set_title(ylabel)

def create_area_chart(dashboard, title, values_r, values_m, column_index, row_index):
    # Create the area chart
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    # Calculate the density for class R
    density_r = gaussian_kde(values_r)
    x_vals_r = np.linspace(min(values_r), max(values_r), 100)
    y_vals_r = density_r(x_vals_r)

    # Add the area series for class R
    area_series_r = chart.add_positive_area_series()
    area_series_r.add(x_vals_r.tolist(), y_vals_r.tolist())
    area_series_r.set_name('R')

    # Calculate the density for class M
    density_m = gaussian_kde(values_m)
    x_vals_m = np.linspace(min(values_m), max(values_m), 100)
    y_vals_m = density_m(x_vals_m)

    # Add the area series for class M
    area_series_m = chart.add_positive_area_series()
    area_series_m.add(x_vals_m.tolist(), y_vals_m.tolist())
    area_series_m.set_name('M')

    # Set axis labels
    chart.get_default_x_axis().set_title('Value')
    chart.get_default_y_axis().set_title('Density')

    # Add a legend
    chart.add_legend()

# Create scatter plots and area charts for each pair and add them to the dashboard
for row_index, y_col in enumerate(columns):
    for column_index, x_col in enumerate(columns):
        if row_index == column_index:
            # Create an area chart on the diagonal
            values_r = sonar_data[sonar_data[target_col] == 'R'][y_col].astype(float).tolist()
            values_m = sonar_data[sonar_data[target_col] == 'M'][y_col].astype(float).tolist()
            title = f'Density of Feature {y_col}'
            create_area_chart(dashboard, title, values_r, values_m, column_index, row_index)
        else:
            # Create a scatter plot for off-diagonal elements
            x_values = sonar_data[x_col].astype(float).tolist()
            y_values = sonar_data[y_col].astype(float).tolist()
            title = f'{x_col} vs {y_col}'
            create_scatter_chart(dashboard, title, x_values, y_values, x_col, y_col, column_index, row_index)

# Open the dashboard in the default web browser
dashboard.open()
