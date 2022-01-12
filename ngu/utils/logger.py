import os
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, env_name, log_csv_path, log_dir, split='train'):
        self.env_name = env_name
        self.log_csv_path = log_csv_path
        self.split = split
        self.ep = 0
        self.ep_rewards = []
        self.losses = []

        self.log_dir = os.path.join(log_dir, env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, timestep):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, timestep)
