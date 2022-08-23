import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from collections import namedtuple
from typing import Any

class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data



device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

tb_writer = SummaryWriter(log_dir="runs/flower_experiment")

model = resnet50_fpn_backbone(returned_layers=[1, 2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3).to(device)
model.eval()
model_wrapper = ModelWrapper(model)
init_img = torch.zeros((1, 3, 500, 500), device=device)
tb_writer.add_graph(model_wrapper, init_img)