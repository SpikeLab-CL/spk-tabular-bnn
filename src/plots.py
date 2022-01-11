from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

red = "#E07182"
blue = "#4487D3"
green = "#96D6B4"
purple = "#B140C8"
grey = "#87878A"

COLOR_DICT = {
    'train': red, 'test': blue, 'valid': green
}
METRIC_NAMES = {
    'elbo': 'ELBO', 
    'mape': 'MAPE',
    'mae': 'MAE',
    'diameter_80pct': 'Percentual diameter 80%', 
    'diameter_80pct_abs': 'Absolute diameter 80%',
    'confidence_error': 'Confidence Error',
}

def plot_metrics(metrics):
    
    fig_list = list()
    for metric_name in METRIC_NAMES.keys():
        fig = go.Figure()
        for partition in ['train', 'test', 'valid']:
            df = metrics.query(f'metric == "{metric_name}" and partition == "{partition}"')
            fig.add_trace(
                go.Scatter(
                    x=df["iteration"], y=df["value"], 
                    name=partition, 
                    line_color=COLOR_DICT[partition],
                )
            )
        fig.update_layout(xaxis_title="Iteration", title=METRIC_NAMES[metric_name])
        fig_list.append(fig) 
    return fig_list

def plot_coverage_curves(coverage_curves):
    fig = go.Figure()
    for partition in coverage_curves.columns:
        fig.add_trace(
            go.Scatter(
                x=coverage_curves.index,
                y=coverage_curves[partition],
                line_color=COLOR_DICT[partition],
                name=partition,
            )
        )
        
    x_grid = np.linspace(min(coverage_curves.index), max(coverage_curves.index), 100)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=x_grid,
            line_color='black',
            name='theoretical',
           # mode='dash',
        )
    )
    fig.update_layout(
        title="Theoretical and Emprical Confidences",
        xaxis_title="Confidence level",
        yaxis_title="Coverage",
    )
    
    return fig