from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
import numpy as np

class Log:
    def __init__(self, batch_size:int, split:str, log_dir:str = 'runs/')->None:
        self.batch_size=batch_size
        self.log_dir=log_dir
        self.split=split
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}{self.split}-{time.strftime("%Y%m%d-%H%M")}', filename_suffix=f'{time.strftime("%Y%m%d-%H%M%S")}')

    def log_metrics(self, scalar:list, epoca:int, scalar_name:str):
        self.writer.add_scalar(f'{scalar_name}', np.mean(scalar), epoca)
        self.writer.flush()
    def log_hiper(self, scalar:list, epoca:int, scalar_name:str):
        self.writer.add_scalar(f'HIPER/{scalar_name}', scalar, epoca)
        self.writer.flush()
    def log_image(self, images, epoca:int, path:str=None):
      img_grid = make_grid(images, nrow=self.batch_size)
      self.writer.add_image(path, img_grid,global_step=epoca)

    def close(self):
        self.writer.close()
