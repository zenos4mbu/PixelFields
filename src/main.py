import os
import numpy as np
import warnings
import torch
import pytorch_lightning as pl

from torchvision.transforms import Compose, ToTensor

from pytorch_lightning.loggers import WandbLogger

from models import SimpleModel

import flags
from absl import app
from absl import flags
# from modules import batch_generator
FLAGS = flags.FLAGS

from datamodules import ImageDataModule

def init_all():
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True

    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()

def main(argv):
    init_all()
    # wandb_logger = WandbLogger(project="PixelFields", name=FLAGS.exp_name)
    # wandb_logger.experiment.config.update(FLAGS)

    dm = ImageDataModule(FLAGS)

    model = SimpleModel(FLAGS)
    # wandb_logger.watch(model, log='all')

    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.load_checkpoint_dir)

    trainer = pl.Trainer(
        gpus=-1, 
        strategy='dp', 
        resume_from_checkpoint=checkpoint_dir if FLAGS.resume_training else None, 
        max_epochs=FLAGS.max_epochs, 
        # logger=wandb_logger,
        # gradient_clip_val=0.5,
        precision=16,
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    app.run(main)