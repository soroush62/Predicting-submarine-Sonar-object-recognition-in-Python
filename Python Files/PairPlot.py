import numpy as np
import pandas as pd
import lightningchart as lc
from scipy.stats import gaussian_kde

lc.set_license('my-license-key')

file_path = 'sonar.csv'
data = pd.read_csv(file_path, header=None)

X = data.drop(columns=[60])
y = data[60]

data['Target'] = y

dashboard = lc.Dashboard(
    rows=len(X.columns[:4]),
    columns=len(X.columns[:4]),
    theme=lc.Themes.Dark
)

colors = {
    'R': lc.Color(30, 144, 255, 128),  # Blue with opacity
    'M': lc.Color(255, 165, 0, 128)    # Orange with opacity
}

def create_density_chart(dashboard, title, values_r, values_m, column_index, row_index):

    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    density_r = gaussian_kde(values_r)
    density_m = gaussian_kde(values_m)
    x_vals = np.linspace(min(values_r.min(), values_m.min()), max(values_r.max(), values_m.max()), 100)
    y_vals_r = density_r(x_vals)
    y_vals_m = density_m(x_vals)

    series_r = chart.add_area_series()
    series_r.add(x_vals.tolist(), y_vals_r.tolist())
    series_r.set_name('R')
    series_r.set_fill_color(lc.Color(30, 144, 255, 128))

    series_m = chart.add_area_series()
    series_m.add(x_vals.tolist(), y_vals_m.tolist())
    series_m.set_name('M')
    series_m.set_fill_color(lc.Color(255, 165, 0, 128))

    legend=chart.add_legend()
    legend.add(data=series_r)
    legend.add(data=series_m)
    chart.get_default_x_axis().set_title('Value')
    chart.get_default_y_axis().set_title('Density')

def create_scatter_chart(dashboard, title, x_values_r, y_values_r, x_values_m, y_values_m, xlabel, ylabel, column_index, row_index):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    scatter_series_r = chart.add_point_series()
    scatter_series_r.add(x_values_r, y_values_r)
    scatter_series_r.set_point_color((lc.Color(30, 144, 255)))

    scatter_series_m = chart.add_point_series()
    scatter_series_m.add(x_values_m, y_values_m)
    scatter_series_m.set_point_color((lc.Color(255, 165, 0)))

    chart.get_default_x_axis().set_title(xlabel)
    chart.get_default_y_axis().set_title(ylabel)

for row_index, y_col in enumerate(X.columns[:4]):
    for column_index, x_col in enumerate(X.columns[:4]):
        if row_index == column_index:
            values_r = data[data['Target'] == 'R'][x_col].astype(float).tolist()
            values_m = data[data['Target'] == 'M'][x_col].astype(float).tolist()
            title = f'Density of Feature {x_col}'
            create_density_chart(dashboard, title, np.array(values_r), np.array(values_m), column_index, row_index)
        else:
            x_values_r = data[data['Target'] == 'R'][x_col].astype(float).tolist()
            y_values_r = data[data['Target'] == 'R'][y_col].astype(float).tolist()
            x_values_m = data[data['Target'] == 'M'][x_col].astype(float).tolist()
            y_values_m = data[data['Target'] == 'M'][y_col].astype(float).tolist()
            title = f'{x_col} vs {y_col}'
            create_scatter_chart(dashboard, title, x_values_r, y_values_r, x_values_m, y_values_m, str(x_col), str(y_col), column_index, row_index)

dashboard.open()
