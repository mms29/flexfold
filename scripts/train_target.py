
import argparse
import os
import pickle
import sys
import contextlib
import logging
from datetime import datetime as dt
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

try:
    import apex.amp as amp  # type: ignore  # PYR01
except ImportError:
    pass

import cryodrgn
from cryodrgn import __version__, ctf
from cryodrgn.beta_schedule import get_beta_schedule

import cryodrgn.config
from cryodrgn import fft
from cryodrgn.source import write_mrc
from openfold.utils.tensor_utils import tensor_tree_map

from openfold.utils.loss import fape_loss, compute_renamed_ground_truth, supervised_chi_loss, find_structural_violations, violation_loss
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


from flexfold.lattice import Lattice
from flexfold import dataset

from flexfold.models import HetOnlyVAE, AFDecoderReal, AFDecoder, struct_to_crd
from flexfold.pose import PoseTracker
from flexfold.core import vol_real, get_cc, fourier_corr, output_single_pdb, struct_to_pdb
from pytorch_lightning.strategies import DDPStrategy

from openfold.utils.loss import fape_loss, compute_renamed_ground_truth
from torch.utils.data import Dataset, DataLoader

from scripts.train import LitDataModule,LitHetOnlyVAE, save_checkpoint, save_config, add_args
from pytorch_lightning.plugins.environments import MPIEnvironment

logger = logging.getLogger(__name__)



class LitTarget(LitHetOnlyVAE):
    
    def training_step(self, batch):

        # Get optimizers
        if args.do_pose_sgd:
            optimizer, _ = self.optimizers()
        else:
            optimizer = self.optimizers()

        # Set gradients to zero
        optimizer.zero_grad()

        # Get scheduler
        sch = self.lr_schedulers() 

        z_mu = torch.zeros(self.model.decoder.zdim, device=self.device)[None]
        z_logvar = torch.ones(self.model.decoder.zdim, device=self.device)[None]
        z = self.model.reparameterize(z_mu, z_logvar)

        struct = self.model.decoder.structure_decoder(z)

        struct.update(
            compute_renamed_ground_truth(
                struct,
                struct["sm"]["positions"][-1],
            )
        )

        loss = fape_loss(
            out =struct,
            batch = struct,
            config = self.model.decoder.loss_config.fape)
        
        # Backward pass
        self.manual_backward(loss)

        # Optimizer step
        optimizer.step()

        # Scheduler step
        sch.step()
        
        if self.global_step %100 == 0:
            if self.trainer.is_global_zero:
                logger.info("Writing checkpoint at step %s ..."%self.global_step)
                out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.current_epoch)
                out_z = "{}/z.{}.pkl".format(self.args.outdir, self.current_epoch)
                out_pdb = "{}/fit.{}.pdb".format(self.args.outdir, self.current_epoch)

                
                save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu, z_logvar, out_weights, out_z)
                struct_to_pdb(tensor_tree_map(lambda x: (x.float() if x.dtype==torch.bfloat16 else x).detach().cpu().numpy()[-1], struct), out_pdb)

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 100  # Only one element total

    def __getitem__(self, idx):
        return torch.empty(0)


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        # optionally split data if needed; skip if you only have train
        self.train_data = DummyDataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1)
    
def main(args: argparse.Namespace) -> None:


    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.overwrite:
        os.system("rm -rvf %s/*"%args.outdir )

    pl.seed_everything(args.seed)

    # load dataset ------------------------------------------------------------------------------------------------------------------------
    logger.info(f"Loading dataset")
    dummydatamodule = DummyDataModule(args)
    dummydatamodule.setup()
    datamodule = LitDataModule(args)
    datamodule.setup()
    # load model ------------------------------------------------------------------------------------------------------------------------
    model = LitTarget(args, datamodule.imageDataset.D, datamodule.imageDataset.N)

    cluster_environment = MPIEnvironment() if args.mpi_plugin else None

    n_devices = torch.cuda.device_count() if args.devices == "auto" else int(args.devices)

    if n_devices >1:
        strategy = DDPStrategy(find_unused_parameters=True,
                               static_graph=True,
                                cluster_environment=cluster_environment,
                                process_group_backend="nccl")
                                # process_group_backend="gloo")
    else:
        strategy="auto"


    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        strategy=strategy,
        # strategy=DDPStrategy(),
        devices=args.devices,                 # or >1 for multi-GPU
        precision=args.precision,  # AMP support
        log_every_n_steps=10,
        callbacks=None,           # optional callbacks like ModelCheckpoint
        logger=CSVLogger(args.outdir, name="", version="")
    )

    trainer.fit(model, datamodule=dummydatamodule, ckpt_path=args.load)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)