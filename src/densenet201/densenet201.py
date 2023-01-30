import torch
import torch.nn as nn

def get_model(pretrained:bool = True, requires_grad:bool = True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=pretrained)
    
    for params in model.parameters():
        params.requires_grad = requires_grad

    model.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)

    return model