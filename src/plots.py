from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd

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
    'MAPE': 'MAPE',
    'MAE': 'MAE',
    'emp_coverage_80pct': 'Empirical coverage for 80%',
    'perc_lenght_80pct': 'Percentual length 80%', 
    'abs_lenght_80pct': 'Absolute length 80%',
    'confidence_error': 'Confidence Error',
}

def add_title_and_axis(
    fig,
    xaxis_title: str=None,
    yaxis_title: str=None,
    title: str=None
):
    if xaxis_title is None and yaxis_title is None and title is None:
        pass
    else:
        fig.update_layout(
            yaxis=dict(title=yaxis_title),
            xaxis=dict(title=xaxis_title),
            title=title,
            hovermode="x",
            hoverlabel_align='right',
        ) 

def plot_lines(
    data: pd.DataFrame, 
    line_config: list, 
    index_col: str=None,
    xaxis_title: str=None,
    yaxis_title: str=None,
    title: str=None,
):
    fig_list = []
    for line in line_config:
        fig_list.append(
            go.Scatter(
                name=line["label"] if "label" in line.keys() else None,
                x=data[index_col] if index_col is not None else data.index,
                y=data[line["column"]],
                mode='lines',
                line=dict(color=line["color"], dash=line["dash"] if "dash" in line.keys() else None),
                showlegend=line["showlegend"] if "showlegend" in line.keys() else True,

            )        
        )
    fig = go.Figure(fig_list)
    add_title_and_axis(fig, xaxis_title, yaxis_title, title)
    return fig
    
def plot_areas(
    data: pd.DataFrame, 
    area_config: list, 
    index_col: str=None,
    xaxis_title: str=None,
    yaxis_title: str=None,
    title: str=None,
):
    fig_list = []
    x = data[index_col].values if index_col is not None else data.index.values
    x = np.concatenate([x, x[::-1]]) #we need to add the reverse of the index
    
    for area in area_config:
        y = np.concatenate([data[area["upper_column"]].values, data[area["lower_column"]].values[::-1]])
        fig_list.append(
            go.Scatter(
                name=area["label"] if "label" in area.keys() else None,
                x=x,
                y=y,
                fill='toself',
                fillcolor=area["color"],
                line_color='rgba(255,255,255,0)',
                showlegend=area["showlegend"] if "showlegend" in area.keys() else True,

            )
        )
    fig = go.Figure(fig_list)
    add_title_and_axis(fig, xaxis_title, yaxis_title, title)
    return fig

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