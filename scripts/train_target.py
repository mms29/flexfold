"""Train a VAE for heterogeneous reconstruction with known poses.

Example usage
-------------
$ cryodrgn train_vae projections.mrcs -o outs/002_trainvae --lr 0.0001 --zdim 8 \
                                      --poses angles.pkl --ctf test_ctf.pkl -n 25

# Restart after already running the same command with some epochs completed
$ cryodrgn train_vae projections.mrcs -o outs/002_trainvae --lr 0.0001 --zdim 8 \
                                      --poses angles.pkl --ctf test_ctf.pkl \
                                      --load latest -n 50

# cryoDRGN-ET tilt series reconstruction
$ cryodrgn train_vae particles_from_M.star --datadir particleseries -o your-outdir \
                                           --ctf ctf.pkl --poses pose.pkl \
                                           --encode-mode tilt --dose-per-tilt 2.93 \
                                           --zdim 12 --num-epochs 50 --beta .025

"""
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



logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable"
    )

    parser.add_argument("--embedding_path",type=os.path.abspath,required=True, help="TODO",
    )
    parser.add_argument("--af_checkpoint_path",type=os.path.abspath,required=True, help="TODO",
    )
    parser.add_argument(
        "--initial_pose_path",type=os.path.abspath,required=True,help="TODO",
    )
    parser.add_argument(
        "--pixel_size", type=float, required=True, help="TODO"
    )
    parser.add_argument(
        "--sigma", type=float, default=2.5, help="TODO"
    )
    parser.add_argument(
        "--af_decoder", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--real_space", action="store_true", help="TODO"
    )    
    parser.add_argument(
        "--quality_ratio", type=float, default=5.0, help="TODO"
    )
    parser.add_argument(
        "--data_loss_weight", type=float, default=1.0, help="TODO"
    )    
    parser.add_argument(
        "--center_loss_weight", type=float, default=0.01, help="TODO"
    )
    parser.add_argument(
        "--viol_loss_weight", type=float, default=1.0, help="TODO"
    )
    parser.add_argument(
        "--chi_loss_weight", type=float, default=1.0, help="TODO"
    )
    parser.add_argument(
        "--all_atom", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--multimer", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--debug", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="TODO"
    )
    parser.add_argument(
        "--pair_stack", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--target_file", type=str, default="auto", help="TODO"
    )
    parser.add_argument(
        "--poses", type=os.path.abspath, required=True, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--load", metavar="WEIGHTS.PKL", help="Initialize training from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1,
        help="Checkpointing interval in N_EPOCHS (default: %(default)s)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Logging interval in N_IMGS (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase verbosity"
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, 100000), help="Random seed"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for data shuffling",
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--ind",
        type=os.path.abspath,
        metavar="PKL",
        help="Filter particles by these indices",
    )
    group.add_argument(
        "--uninvert-data",
        dest="invert_data",
        action="store_false",
        help="Do not invert data sign",
    )
    group.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="Turn off real space windowing of dataset",
    )
    group.add_argument(
        "--window-r",
        type=float,
        default=0.85,
        help="Windowing radius (default: %(default)s)",
    )
    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    group.add_argument(
        "--shuffler-size",
        type=int,
        default=0,
        help="If non-zero, will use a data shuffler for faster lazy data loading.",
    )
    group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses to use as DataLoader workers. If 0, then use the main process for data loading. (default: %(default)s)",
    )
    group.add_argument(
        "--max-threads",
        type=int,
        default=16,
        help="Maximum number of CPU cores for data loading (default: %(default)s)",
    )

    group = parser.add_argument_group("Tilt series parameters")
    group.add_argument(
        "--ntilts",
        type=int,
        default=10,
        help="Number of tilts to encode (default: %(default)s)",
    )
    group.add_argument(
        "--random-tilts",
        action="store_true",
        help="Randomize ordering of tilts series to encoder",
    )
    group.add_argument(
        "--t-emb-dim",
        type=int,
        default=64,
        help="Intermediate embedding dimension (default: %(default)s)",
    )
    group.add_argument(
        "--tlayers",
        type=int,
        default=3,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--tdim",
        type=int,
        default=1024,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "-d",
        "--dose-per-tilt",
        type=float,
        help="Expected dose per tilt (electrons/A^2 per tilt) (default: %(default)s)",
    )
    group.add_argument(
        "-a",
        "--angle-per-tilt",
        type=float,
        default=3,
        help="Tilt angle increment per tilt in degrees (default: %(default)s)",
    )

    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Minibatch size (default: %(default)s)",
    )
    group.add_argument(
        "--wd",
        type=float,
        default=0,
        help="Weight decay in Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate in Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--beta",
        default=None,
        help="Choice of beta schedule or a constant for KLD weight (default: 1/zdim)",
    )
    group.add_argument(
        "--beta-control",
        type=float,
        help="KL-Controlled VAE gamma. Beta is KL target",
    )
    group.add_argument(
        "--norm",
        type=float,
        nargs=2,
        default=None,
        help="Data normalization as shift, 1/scale (default: mean, std of dataset)",
    )
    group.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="TODO",
    )
    group.add_argument(
        "--multigpu",
        action="store_true",
        help="Parallelize training across all detected GPUs",
    )

    group = parser.add_argument_group("Pose SGD")
    group.add_argument(
        "--do-pose-sgd", action="store_true", help="Refine poses with gradient descent"
    )
    group.add_argument(
        "--pretrain",
        type=int,
        default=1,
        help="Number of epochs with fixed poses before pose SGD (default: %(default)s)",
    )
    group.add_argument(
        "--emb-type",
        choices=("s2s2", "quat"),
        default="quat",
        help="SO(3) embedding type for pose SGD (default: %(default)s)",
    )
    group.add_argument(
        "--pose-lr",
        type=float,
        default=3e-4,
        help="Learning rate for pose optimizer (default: %(default)s)",
    )

    group = parser.add_argument_group("Encoder Network")
    group.add_argument(
        "--enc-layers",
        dest="qlayers",
        type=int,
        default=3,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--enc-dim",
        dest="qdim",
        type=int,
        default=1024,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--encode-mode",
        default="resid",
        choices=("conv", "resid", "mlp", "tilt"),
        help="Type of encoder network (default: %(default)s)",
    )
    group.add_argument(
        "--enc-mask",
        type=int,
        help="Circular mask of image for encoder (default: D/2; -1 for no mask)",
    )
    group.add_argument(
        "--use-real",
        action="store_true",
        help="Use real space image for encoder (for convolutional encoder)",
    )

    group = parser.add_argument_group("Decoder Network")
    group.add_argument(
        "--dec-layers",
        dest="players",
        type=int,
        default=3,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--dec-dim",
        dest="pdim",
        type=int,
        default=1024,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--pe-type",
        choices=(
            "geom_ft",
            "geom_full",
            "geom_lowf",
            "geom_nohighf",
            "linear_lowf",
            "gaussian",
            "none",
        ),
        default="gaussian",
        help="Type of positional encoding (default: %(default)s)",
    )
    group.add_argument(
        "--feat-sigma",
        type=float,
        default=0.5,
        help="Scale for random Gaussian features (default: %(default)s)",
    )
    group.add_argument(
        "--pe-dim",
        type=int,
        help="Num frequencies in positional encoding (default: image D/2)",
    )
    group.add_argument(
        "--domain",
        choices=("hartley", "fourier"),
        default="fourier",
        help="Volume decoder representation (default: %(default)s)",
    )
    group.add_argument(
        "--activation",
        choices=("relu", "leaky_relu"),
        default="relu",
        help="Activation (default: %(default)s)",
    )


    return parser



def save_config(args, out_config):
    model_args = dict(
        qlayers=args.qlayers,
        qdim=args.qdim,
        players=args.players,
        pdim=args.pdim,
        zdim=args.zdim,
        encode_mode=args.encode_mode,
        enc_mask=args.enc_mask,
        pe_type=args.pe_type,
        feat_sigma=args.feat_sigma,
        pe_dim=args.pe_dim,
        domain=args.domain,
        activation=args.activation,
        tilt_params=dict(
            tdim=args.tdim,
            tlayers=args.tlayers,
            t_emb_dim=args.t_emb_dim,
            ntilts=args.ntilts,
        ),

        af_decoder = args.af_decoder,
        initial_pose_path = args.initial_pose_path,
        embedding_path=args.embedding_path,
        af_checkpoint_path =args.af_checkpoint_path,
        sigma = args.sigma,
        pixel_size = args.pixel_size,
        quality_ratio = args.quality_ratio,
        all_atom = args.all_atom,
        pair_stack = args.pair_stack,
        target_file= args.target_file,
        real_space = args.real_space,
        is_multimer = args.multimer,
    )
    config = dict(
        model_args=model_args
    )
    config["seed"] = args.seed
    cryodrgn.config.save(config, out_config)


class LitHetOnlyVAE(pl.LightningModule):
    def __init__(self, args, D,Nparticles):
        self.args = args
        self.D = D
        self.Nparticles = Nparticles
        super().__init__()

        if args.encode_mode == "conv":
            assert self.D - 1 == 64, "Image size must be 64x64 for convolutional encoder"


        self.domain = self.args.domain

        # load lattice ------------------------------------------------------------------------------------------------------------------------
        self.lattice = Lattice(D, extent=0.5)
        if args.enc_mask is None:
            args.enc_mask = D // 2
        if args.enc_mask > 0:
            assert args.enc_mask <= D // 2
            enc_mask = self.lattice.get_circular_mask(args.enc_mask)
            in_dim = int(enc_mask.sum())
        elif args.enc_mask == -1:
            enc_mask = None
            in_dim = self.lattice.D**2 if not args.use_real else (self.lattice.D - 1) ** 2
        else:
            raise RuntimeError(
                "Invalid argument for encoder mask radius {}".format(args.enc_mask)
            )
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]

        # load poses ------------------------------------------------------------------------------------------------------------------------
        if args.do_pose_sgd:
            assert (
                args.domain == "hartley"
            ), "Need to use --domain hartley if doing pose SGD"
        do_pose_sgd = args.do_pose_sgd
        self.posetracker = PoseTracker.load(
            args.poses, Nparticles, D, "s2s2" if do_pose_sgd else None, args.ind,
        )

        # load ctf ------------------------------------------------------------------------------------------------------------------------
        if args.ctf is not None:
            if args.use_real:
                raise NotImplementedError(
                    "Not implemented with real-space encoder. Use phase-flipped images instead"
                )
            logger.info("Loading ctf params from {}".format(args.ctf))
            ctf_params = ctf.load_ctf_for_training(D - 1, args.ctf)
            if args.ind is not None:
                ctf_params = ctf_params[args.ind, ...]
            assert ctf_params.shape == (Nparticles, 8)
            self.register_buffer("ctf_params", torch.tensor(ctf_params))  # Nx8
        else:
            self.ctf_params = None

        # load beta ------------------------------------------------------------------------------------------------------------------------
        if args.beta is None:
            args.beta = 1.0 / args.zdim
        try:
            args.beta = float(args.beta)
        except ValueError:
            assert (
                args.beta_control
            ), "Need to set beta control weight for schedule {}".format(args.beta)
        self.beta_schedule = get_beta_schedule(args.beta)

        # load model ------------------------------------------------------------------------------------------------------------------------
        self.model = HetOnlyVAE(
            self.lattice,
            args.qlayers,
            args.qdim,
            args.players,
            args.pdim,
            in_dim,
            args.zdim,
            encode_mode=args.encode_mode,
            enc_mask=enc_mask,
            enc_type=args.pe_type,
            enc_dim=args.pe_dim,
            domain=args.domain,
            activation=activation,
            feat_sigma=args.feat_sigma,
            tilt_params={},
            af_decoder=args.af_decoder,
            initial_pose_path=args.initial_pose_path,
            embedding_path=args.embedding_path,
            sigma=args.sigma,
            pixel_size=args.pixel_size,
            quality_ratio=args.quality_ratio,
            all_atom=args.all_atom,
            pair_stack=args.pair_stack,
            target_file=args.target_file,
            real_space=args.real_space,
            is_multimer = args.multimer,
            af_checkpoint_path =args.af_checkpoint_path,
        )

        self.val_z_mu = []
        self.val_z_logvar = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr,  weight_decay=self.args.wd
        )
        return optimizer
        

    
    def training_step(self, batch):
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
        
        if self.current_epoch%100 == 0:

            out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.current_epoch)
            out_z = "{}/z.{}.pkl".format(self.args.outdir, self.current_epoch)
            out_pdb = "{}/fit.{}.pdb".format(self.args.outdir, self.current_epoch)

            
            save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu, z_logvar, out_weights, out_z)
            struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], struct), out_pdb)

        return loss

def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    """Save model weights, latent encoding z, and decoder volumes"""
    # save model weights
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        out_weights,
    )
    # save z
    with open(out_z, "wb") as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1  # Only one element total

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
    def val_dataloader(self):
        return DataLoader(self.train_data, batch_size=1)

    
def main(args: argparse.Namespace) -> None:


    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.overwrite:
        os.system("rm -rvf %s/*"%args.outdir )

    pl.seed_everything(args.seed)

    # load dataset ------------------------------------------------------------------------------------------------------------------------
    logger.info(f"Loading dataset")
    datamodule = DummyDataModule(args)
    datamodule.setup()

    # load model ------------------------------------------------------------------------------------------------------------------------
    model = LitHetOnlyVAE(args, 128 + 1, 100000)

    # save configuration
    out_config = "{}/config.yaml".format(args.outdir)
    save_config(args,out_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.outdir,          # where to save
        filename="ckpt",  # filename pattern
        monitor="loss",              # what metric to track
        save_top_k=3,                    # save the 3 best models
        mode="min",                      # "min" for loss, "max" for accuracy, etc.
        save_last=False,                  # also save 'last.ckpt'
        verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        strategy="auto",
        # strategy=DDPStrategy(),
        devices=1,                 # or >1 for multi-GPU
        precision=args.precision,  # AMP support
        log_every_n_steps=10,
        callbacks=checkpoint_callback,           # optional callbacks like ModelCheckpoint
        logger=CSVLogger(args.outdir, name="", version="")
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.load)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)