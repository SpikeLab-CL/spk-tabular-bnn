import numpy as np
import pandas as pd
from bnn_utils import get_predictions_pyro

def get_empiric_coverage_curve(
    valid_post,
    data,
    target,
    inv_transf,
    coverages=np.linspace(.001, .999, 50)
):

    perc = np.concatenate([(1-coverages) / 2, (1+coverages) / 2])
    quantiles = inv_transf(np.quantile(valid_post, perc, axis=0)).squeeze()
    lower, upper = quantiles[:len(coverages), :], quantiles[len(coverages):, :]
    in_interval = (
        (data[target].values.reshape(1, -1) >= lower) & (data[target].values.reshape(1, -1) <= upper)
    )
    in_interval_df = pd.DataFrame({
        'in_interval': in_interval.flatten(),
        'theoretical_coverage': coverages.repeat(len(data)),
        'partition': data['partition'].values.reshape(1, -1).repeat(len(coverages), axis=0).flatten()
    })
    
    return (
        in_interval_df
        .groupby(['partition', 'theoretical_coverage'])
        .in_interval
        .mean()
        .unstack(0)
    )

def get_metrics(
    model, 
    guide, 
    data, 
    features, 
    target, 
    preprocessor, 
    inv_transf, 
    n_samples=100,
    device='cpu'
):
    
    #sample from aprox posterior and compute quantiles and model average
    valid_post = get_predictions_pyro(
        model,  
        guide,
        preprocessor,
        features,
        data, 
        n_samples=n_samples, 
        varbls=['_RETURN'],
        device=device,
    )['_RETURN']

    coverage_curves = get_empiric_coverage_curve(
        valid_post, 
        data,
        target,
        inv_transf,
        coverages=np.linspace(.001, .999, 50)
    )

    curve_diff = abs(coverage_curves.values - coverage_curves.index.values.reshape((-1, 1)))
    integral = pd.DataFrame(
        (0.5 * (curve_diff[1:, :] + curve_diff[:-1, :]) * np.diff(coverage_curves.index).reshape(-1, 1)).sum(axis=0)*2,
        index=coverage_curves.columns,
        columns=['confidence_error']
    )
    
    q10 = inv_transf(np.quantile(valid_post, 0.1, axis=0))
    q90 = inv_transf(np.quantile(valid_post, 0.9, axis=0))
    mean = inv_transf(valid_post.mean(axis=0).squeeze())
    
    eval_data = data[['partition', target]].copy()
    eval_data[f'{target}_mean'] = mean
    eval_data[f'{target}_q10'] = q10
    eval_data[f'{target}_q90'] = q90
    
    #compute model quality metrics (precision and uncertenty estimation)                 
    mertic_formulas = {
        "coverage_80pct": f"({target} >= {target}_q10) & ({target} <= {target}_q90)",
        "diameter_80pct": f"({target}_q90 - {target}_q10)/{target}",
        "diameter_80pct_abs": f"({target}_q90 - {target}_q10)",
        "mape": f"abs({target}-{target}_mean)/{target}",
        "mae": f"abs({target}-{target}_mean)"
    }                  
    
    for metric, formula in mertic_formulas.items():
        eval_data[metric] = eval_data.eval(formula)
                      
    metrics = eval_data.groupby('partition')[list(mertic_formulas.keys())].apply(lambda x: np.mean(x[x < np.infty]))
    
    return pd.concat([metrics, integral], axis=1)

