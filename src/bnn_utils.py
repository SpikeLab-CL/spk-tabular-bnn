import pyro
import pyro.distributions as dist

from pyro.infer import Predictive
import torch
from pyro.nn.module import to_pyro_module_
from pyro.nn.module import PyroSample

#I'm not retraining the embeddings and batch norm on the bayesian opt
FORBIDDEN_NAMES = ["embeddings", "bn.bn"]

def as_pyro_module(
    module, 
    scale=1.0, 
    tabnet_centered=True, 
    forbidden_names=FORBIDDEN_NAMES,
):
    to_pyro_module_(module, recurse=True)
    for module_name, sub_module in module.named_modules():
        if all(not name in module_name for name in forbidden_names):
            for param_name, param in list(sub_module.named_parameters(recurse=False)):
                
                scale_tensor = scale*torch.ones_like(param)
                if tabnet_centered:
                    mean_tensor = param
                else:
                    mean_tensor = torch.zeros_like(param)
                setattr(
                        sub_module, 
                        param_name, 
                        PyroSample(dist.Normal(mean_tensor, scale_tensor).to_event())
                    )
        else:
            for param_name, param in list(sub_module.named_parameters(recurse=False)):
                setattr(sub_module, param_name, param)
    return module

def get_predictions_pyro(
    model, 
    guide,
    preprocessor,
    features,
    df, 
    n_samples=100, 
    varbls=['_RETURN'],
    device='cpu',
):
    
    x = torch.Tensor(preprocessor.transform(df[features])).to(device)
    
    predictive = Predictive(model, guide=guide, num_samples=n_samples, return_sites=varbls)
    samples_post = {v: predictive(x, None)[v].cpu().detach().numpy() for v in varbls}
    
    return samples_post