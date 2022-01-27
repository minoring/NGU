from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, env_id, log_dir, split='train'):
        self.env_id = env_id
        self.log_dir = log_dir
        self.split = split
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, name, scalar, timestep):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, timestep)
