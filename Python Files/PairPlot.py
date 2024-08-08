import numpy as np
import pandas as pd
import lightningchart as lc
from scipy.stats import gaussian_kde

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

# Add the target variable to the dataset for visualization
data['Target'] = y

# Create a Dashboard
dashboard = lc.Dashboard(
    rows=len(X.columns[:4]),
    columns=len(X.columns[:4]),
    theme=lc.Themes.Dark
)

# Define colors for the different target classes
colors = {
    'R': lc.Color(30, 144, 255, 128),  # Blue with opacity
    'M': lc.Color(255, 165, 0, 128)    # Orange with opacity
}

def create_density_chart(dashboard, title, values_r, values_m, column_index, row_index):
    # Create the density chart
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    # Calculate density
    density_r = gaussian_kde(values_r)
    density_m = gaussian_kde(values_m)
    x_vals = np.linspace(min(values_r.min(), values_m.min()), max(values_r.max(), values_m.max()), 100)
    y_vals_r = density_r(x_vals)
    y_vals_m = density_m(x_vals)

    # Add density areas
    series_r = chart.add_area_series()
    series_r.add(x_vals.tolist(), y_vals_r.tolist())
    series_r.set_name('R')
    series_r.set_fill_color(lc.Color(30, 144, 255, 128))

    series_m = chart.add_area_series()
    series_m.add(x_vals.tolist(), y_vals_m.tolist())
    series_m.set_name('M')
    series_m.set_fill_color(lc.Color(255, 165, 0, 128))

    chart.add_legend()
    chart.get_default_x_axis().set_title('Value')
    chart.get_default_y_axis().set_title('Density')

def create_scatter_chart(dashboard, title, x_values_r, y_values_r, x_values_m, y_values_m, xlabel, ylabel, column_index, row_index):
    # Create the scatter plot chart
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    # Add the points to the scatter chart
    scatter_series_r = chart.add_point_series()
    scatter_series_r.add(x_values_r, y_values_r)
    scatter_series_r.set_point_color((lc.Color(30, 144, 255)))

    scatter_series_m = chart.add_point_series()
    scatter_series_m.add(x_values_m, y_values_m)
    scatter_series_m.set_point_color((lc.Color(255, 165, 0)))

    # Set axis labels
    chart.get_default_x_axis().set_title(xlabel)
    chart.get_default_y_axis().set_title(ylabel)

# Create charts for each pair and add them to the dashboard
for row_index, y_col in enumerate(X.columns[:4]):
    for column_index, x_col in enumerate(X.columns[:4]):
        if row_index == column_index:
            # Create density chart on the diagonal
            values_r = data[data['Target'] == 'R'][x_col].astype(float).tolist()
            values_m = data[data['Target'] == 'M'][x_col].astype(float).tolist()
            title = f'Density of Feature {x_col}'
            create_density_chart(dashboard, title, np.array(values_r), np.array(values_m), column_index, row_index)
        else:
            # Create scatter chart for other pairs
            x_values_r = data[data['Target'] == 'R'][x_col].astype(float).tolist()
            y_values_r = data[data['Target'] == 'R'][y_col].astype(float).tolist()
            x_values_m = data[data['Target'] == 'M'][x_col].astype(float).tolist()
            y_values_m = data[data['Target'] == 'M'][y_col].astype(float).tolist()
            title = f'{x_col} vs {y_col}'
            create_scatter_chart(dashboard, title, x_values_r, y_values_r, x_values_m, y_values_m, str(x_col), str(y_col), column_index, row_index)

# Open the dashboard in the default web browser
dashboard.open()
