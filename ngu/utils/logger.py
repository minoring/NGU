import os
import time

from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, env_id, log_csv_path, log_dir, split='train'):
        self.env_id = env_id
        self.log_csv_path = log_csv_path
        self.split = split
        self.ep = 0
        self.ep_rewards = []
        self.losses = []

        self.log_dir = os.path.join(log_dir, env_id + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, timestep):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, timestep)
