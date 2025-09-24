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
from pytorch_lightning.plugins.environments import MPIEnvironment

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
from flexfold.core import vol_real, get_cc, fourier_corr, output_single_pdb, struct_to_pdb,weighted_normalized_l2, gaussian_weight, frequency_weights
from pytorch_lightning.strategies import DDPStrategy
from scipy.ndimage import gaussian_filter
from flexfold.core import ifft2_center, unsymmetrize_ht

def pad_to_max(tensor, max_size):
    pad_size = max_size - tensor.size(0)
    if pad_size > 0:
        padding = (0, 0) * (tensor.dim() - 1) + (0, pad_size)  # pad last dim only
        tensor = F.pad(tensor, padding, value=-1)
    return tensor

def print_model_summary(model):
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
        else:
            frozen_params += n

    logger.info(f"{'='*50}")
    logger.info(f"Model Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Total parameters   : {total_params:,}")
    logger.info(f"Trainable params   : {trainable_params:,}")
    logger.info(f"Frozen params      : {frozen_params:,}")
    logger.info(f"{'='*50}")

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
        "--real_space", action="store_true", help="TODO"
    )        
    parser.add_argument(
        "--mpi_plugin", action="store_true", help="TODO"
    )      
    parser.add_argument(
        "--use_lma", action="store_true", help="TODO"
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
        "--train_val_ratio", type=float, default=0.9, help="TODO"
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
        "--devices", type=str, default="auto", help="TODO"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="TODO"
    )
    parser.add_argument(
        "--pair_stack", action="store_true", help="TODO"
    )
    parser.add_argument(
        "--target_file", type=str, default=None, help="TODO"
    )
    parser.add_argument(
        "--frozen_structure_module", action="store_true", help="TODO"
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
        choices=("hartley", "fourier", "real"),
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

def save_config(args, dataset, lattice, out_config):
    dataset_args = dict(
        particles=args.particles,
        norm=dataset.norm,
        invert_data=args.invert_data,
        ind=args.ind,
        window=args.window,
        window_r=args.window_r,
        datadir=args.datadir,
        ctf=args.ctf,
        poses=args.poses,
        do_pose_sgd=args.do_pose_sgd,
    )
    if args.encode_mode == "tilt":
        dataset_args["ntilts"] = args.ntilts

    lattice_args = dict(D=lattice.D, extent=lattice.extent, ignore_DC=lattice.ignore_DC)
    model_args = dict(
        qlayers=args.qlayers,
        qdim=args.qdim,
        players=args.players,
        pdim=args.pdim,
        zdim=args.zdim,
        encode_mode=args.encode_mode,
        enc_mask=args.enc_mask,
        pe_type=args.pe_type,
        pe_dim=args.pe_dim,
        domain=args.domain,
        activation=args.activation,
        initial_pose_path = args.initial_pose_path,
        embedding_path=args.embedding_path,
        af_checkpoint_path =args.af_checkpoint_path,
        sigma = args.sigma,
        pixel_size = args.pixel_size,
        quality_ratio = args.quality_ratio,
        all_atom = args.all_atom,
        pair_stack = args.pair_stack,
        real_space = args.real_space,
        is_multimer = args.multimer,
        target_file= args.target_file,
    )
    config = dict(
        dataset_args=dataset_args, lattice_args=lattice_args, model_args=model_args
    )
    config["seed"] = args.seed
    cryodrgn.config.save(config, out_config)



class LitHetOnlyVAE(pl.LightningModule):
    def __init__(self, args, D,Nparticles):
        self.args = args
        self.D = D
        self.Nparticles = Nparticles
        super().__init__()

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
            in_dim = self.lattice.D**2 if not (args.domain =="real") else (self.lattice.D - 1) ** 2
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
            enc_mask=None if self.domain == "real" else enc_mask,
            enc_type=args.pe_type,
            enc_dim=args.pe_dim,
            domain=args.domain,
            activation=activation,
            initial_pose_path=args.initial_pose_path,
            embedding_path=args.embedding_path,
            sigma=args.sigma,
            pixel_size=args.pixel_size,
            quality_ratio=args.quality_ratio,
            all_atom=args.all_atom,
            pair_stack=args.pair_stack,
            real_space=args.real_space,
            is_multimer = args.multimer,
            af_checkpoint_path =args.af_checkpoint_path,
            target_file=args.target_file,
        )

        self.val_z_mu = []
        self.val_z_logvar = []
        self.val_z_idx = []

        if self.args.frozen_structure_module:
            for param in self.model.decoder.structure_module.parameters():
                param.requires_grad = False

        if self.domain == "real":
            self.model.enc_mask = None

        self.model.decoder.globals.use_lma = args.use_lma

        print_model_summary(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.lr,  weight_decay=self.args.wd
        )
        if self.args.do_pose_sgd : 
            pose_optimizer = torch.optim.SparseAdam(list(self.posetracker.parameters()), lr=self.args.pose_lr)
            return [optimizer, pose_optimizer]
        
        else:
            return optimizer
        
    def prepare_batch(self, batch):
        particles, particles_real, ind = batch
        B = ind.size(0)
        D = self.lattice.D

        # Pose
        rot, tran = self.posetracker.get_pose(ind)

        # Image
        y_real = self.lattice.translate_real(particles_real, tran.unsqueeze(1)).view(B, D-1, D-1)
        y_real = y_real.transpose(-1,-2)

        if self.domain == "fourier":
            y = torch.view_as_real(particles)
            y = self.lattice.translate_ft(y.view(B, D*D, 2), tran.unsqueeze(1)).view(B, D, D, 2)
            # y_real = ifft2_center(torch.view_as_complex(y)).real
        elif self.domain == "hartley":
            y = particles
            y = self.lattice.translate_ht(y.view(B, -1), tran.unsqueeze(1)).view(B, D, D)
            # y_real = fft.iht2_center(y)
        elif self.domain == "real":
            y=y_real.contiguous()
        
        # CTF
        if self.ctf_params is not None:
            ctf_param = self.ctf_params[ind]
            freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                B, *self.lattice.freqs2d.shape
            ) / ctf_param[:, 0].view(B, 1, 1)
            c = ctf.compute_ctf(freqs, *torch.split(ctf_param[:, 1:], 1, 1)).view(
                B, D, D
            )
        else: 
            c =None

        return y, y_real, rot, tran, c
    
    def write_debug(self, struct, mask, y, y_real, y_recon, y_recon_real, global_it):
        """
        y : ?
        y_real :[B, D-1, D-1] Real

        y_recon : [B, D, D] FT complex
        y_recon_real :[B, D-1, D-1] Real
        """
        logger.info("Writing debug PNG at iteration %i"%global_it)
        D = self.lattice.D
        B=y_real.size(0)
        if B> 8 : 
            B=8
        fig, ax = plt.subplots(3, B, layout="constrained", figsize=(5*B,10))
        ax = np.array(ax).reshape(3, B) 
        masked_overlay = np.zeros((D* D, 4))
        masked_overlay[(mask == 0).detach().cpu().numpy()] = [0, 0, 0, 1]   # Black with full opacity
        masked_overlay[(mask == 1).detach().cpu().numpy()] = [0, 0, 0, 0]   # Fully transparent
        masked_overlay = masked_overlay.reshape((D,D,4))

        for i in range(B):
            #filter
            y_ = y_real[i].detach().cpu().numpy()
            r_ = y_recon_real[i].detach().cpu().numpy()

            sigma = 1.5
            y_ = gaussian_filter(y_, sigma=sigma)
            r_ = gaussian_filter(r_, sigma=sigma)
            # Normalize
            def peak_normalize(img):
                m = np.max(np.abs(img), axis=(-2,-1), keepdims=True)
                return img / (m + 1e-8)
            
            def l2_normalize(img):
                norm = np.linalg.norm(img, axis=(-2, -1), keepdims=True)  
                return img / (norm + 1e-8)

            y_ =l2_normalize(y_)
            r_ =l2_normalize(r_)

            vmin= min(y_.min(), r_.min())
            vmax= min(y_.max(), r_.max())
            ax[0,i].imshow(y_, vmin=vmin, vmax=vmax, cmap="Greys")
            ax[1,i].imshow(r_, vmin=vmin, vmax=vmax, cmap="Greys")
            ax[2,i].imshow(np.abs(r_-y_), vmin=0.0, vmax=max(np.abs([vmax,vmin])), cmap="jet")

            # ax[0,i].imshow(masked_overlay)
            # ax[1,i].imshow(masked_overlay)
            # ax[2,i].imshow(masked_overlay)
            ax[0,i].axis('off')
            ax[1,i].axis('off')
            ax[2,i].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(self.args.outdir + "/debug_%s.png"%str(global_it).zfill(5))
        plt.close(fig)
        output_single_pdb(
            all_atom_positions= (struct["final_atom_positions"] @ self.model.decoder.rot_init + self.model.decoder.trans_init[..., None, :]).detach().cpu().numpy()[-1],
            all_atom_mask=struct["all_atom_mask"].detach().cpu().numpy()[-1],
            aatype=struct["aatype"].detach().cpu().numpy()[-1],
            residue_index=struct["residue_index"].detach().cpu().numpy()[-1],
            chain_index=struct["asym_id"].detach().cpu().numpy()[-1] if "asym_id" in struct else None,
            file= self.args.outdir + "/debug_%s.pdb"%str(global_it).zfill(5)
        )
    def run_encoder(self, y, c=None):
        if self.domain == "fourier":
            y = (y[...,0] - y[...,1]) 
        input_ = (y,)
        if self.domain != "real":
            if c is not None:
                input_ = (x * c.sign() for x in input_)  # phase flip by the ctf

        z_mu, z_logvar = self.model.encode(*input_)

        return z_mu, z_logvar 

    def run_decoder(self, z, rot, c):
        B = z.size(0)
        D = self.lattice.D
        mask = self.lattice.get_circular_mask(self.lattice.D // 2)  # restrict to circular mask

        # decode
        if isinstance(self.model.decoder, AFDecoderReal):
            y_recon, struct = self.model(rot, z)
            y_recon =  y_recon.view(B, -1)[..., mask]
        elif isinstance(self.model.decoder, AFDecoder):
            y_recon, struct = self.model(self.lattice.coords[mask] / self.lattice.extent / 2 @ rot, z)
        else:
            raise RuntimeError("Unknown decoder type:%s"%( type(self.model.decoder).__name__))

        # Apply CTF
        y_recon = y_recon.view(B, -1)
        y_recon *= c.view(B, -1)[:, mask]

        # Pad mask with 0
        tmp = y_recon
        y_recon = torch.zeros((B, D*D), dtype=y_recon.dtype, device=y_recon.device)
        y_recon[:, mask] = tmp[:]

        # Real space
        y_recon_real = y_recon.reshape(B, D,D)
        y_recon_real = unsymmetrize_ht(y_recon_real)
        y_recon_real = ifft2_center(y_recon_real).real

        return y_recon, y_recon_real, mask, struct


    def training_step(self, batch, batch_idx):

       
        # Preprocessing
        y, y_real, rot, _, c = self.prepare_batch(batch)

        B = batch[-1].size(0)
        global_it = self.Nparticles * self.current_epoch +  batch_idx * B
        beta = self.beta_schedule( global_it)

        # Encdoer
        z_mu, z_logvar = self.run_encoder(y, c)

        # Reparametrize latent space
        z = self.model.reparameterize(z_mu, z_logvar)

        # Decoder
        y_recon,y_recon_real, mask, struct = self.run_decoder(z,rot, c)

        # Write debug
        if self.args.debug and self.trainer.is_global_zero and ((global_it)%100 == 0  or batch_idx ==0):
            self.write_debug(struct, mask, y, y_real, y_recon, y_recon_real, global_it)

        # Computing loss
        loss, gen_loss, kld = self.loss_function(
            z_mu,
            z_logvar,
            y_real,
            y_recon_real,
            mask,
            beta,
            struct=struct
        )

        # Logging
        self.log("loss", loss, prog_bar=True, sync_dist=False, on_epoch=True, on_step=True)
        self.log("kld", kld, prog_bar=False, sync_dist=False, on_epoch=True, on_step=True)
        for k,v in gen_loss.items():
            self.log(k, v, prog_bar=False, sync_dist=False, on_epoch=True, on_step=True)

        return loss
    
    def on_after_backward(self):
        unused = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                unused.append(name)
        if unused:
            print("⚠️ Unused parameters detected:")
            for u in unused:
                print(u)
            print()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch[-1]
        y, y_real, rot, _, c = self.prepare_batch(batch)
        z_mu, z_logvar = self.run_encoder(y, c)

        self.val_z_mu.append(z_mu)
        self.val_z_logvar.append(z_logvar)
        self.val_z_idx.append(idx)

        if dataloader_idx == 0 :
                _,y_recon_real, _, struct = self.run_decoder(z_mu,rot, c)
                gen_loss, total_gen_loss = self.gen_loss(y_real, y_recon_real, struct)

                self.log("val_loss", total_gen_loss, prog_bar=False, sync_dist=True, on_epoch=True, on_step=True, add_dataloader_idx=False)
                for k,v in gen_loss.items():
                    self.log("val_" + k, v, prog_bar=False, sync_dist=True, on_epoch=True, on_step=True, add_dataloader_idx=False)



    def on_validation_epoch_end(self):

        # Stack tensors from local GPU
        z_mu = torch.cat(self.val_z_mu, dim=0)
        z_logvar = torch.cat(self.val_z_logvar, dim=0)
        z_idx = torch.cat(self.val_z_idx, dim=0)

        batch_size = torch.tensor(z_mu.size(0), device=z_mu.device)
        max_size = self.all_gather(batch_size).max()

        z_mu = pad_to_max(z_mu, max_size)
        z_logvar = pad_to_max(z_logvar, max_size)
        z_idx = pad_to_max(z_idx, max_size)

        # Gather across all GPUs
        z_mu = self.all_gather(z_mu)
        z_logvar = self.all_gather(z_logvar)
        z_idx = self.all_gather(z_idx)

        # Remove padded entries (idx == -1)
        valid_mask = z_idx != -1
        z_mu = z_mu[valid_mask]
        z_logvar = z_logvar[valid_mask]
        z_idx = z_idx[valid_mask]

        # Reorder by idx
        def get_first_idx(A):
            unique, idx, counts = torch.unique(A, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0], device=A.device), cum_sum[:-1]))
            return ind_sorted[cum_sum]
        sorted_idx = get_first_idx(z_idx)
        z_mu = z_mu[sorted_idx]
        z_logvar = z_logvar[sorted_idx]
        z_idx = z_idx[sorted_idx]

        assert  torch.all(z_idx[1:]>z_idx[:-1]), "not continuous!"

        self.val_z_mu.clear()
        self.val_z_logvar.clear() 
        self.val_z_idx.clear() 

        if self.trainer.is_global_zero: 
            out_z = "{}/z.{}.pkl".format(self.args.outdir, self.current_epoch)
            out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.current_epoch)
            save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu.detach().cpu().numpy(), z_logvar.detach().cpu().numpy(), out_weights, out_z)

    def gen_loss(self, y, y_recon, struct):
        # Reconstruction loss
        data_loss = torch.mean(1-get_cc(y, y_recon))

        # Struct violations loss
        struct_violations = find_structural_violations(
            struct,
            struct["sm"]["positions"][-1],
            **self.model.decoder.loss_config.violation,
        )
        viol_loss = violation_loss(
                    struct_violations,
                    **{**struct, **self.model.decoder.loss_config.violation},
                )

        # Torsion angle loss
        chi_loss = supervised_chi_loss(
                    struct["sm"]["angles"],
                    struct["sm"]["unnormalized_angles"],
                    **{**struct, **self.model.decoder.loss_config.supervised_chi},
                )
                
        # Center Loss
        # crd = struct_to_crd(struct, ca=not self.model.decoder.all_atom)
        # crd = crd @ self.model.decoder.rot_init + self.model.decoder.trans_init[..., None, :]
        # center_loss = torch.mean(torch.sum((torch.mean(crd, dim=-2) ** 2 ), dim=-1))
        center_loss = 0.0

        # TOTAL GEN LOSS
        gen_loss = {
            "data_loss": data_loss, 
            "chi_loss" : chi_loss, 
            "viol_loss": viol_loss,
            "center_loss": center_loss,
            }
        total_gen_loss = (
            data_loss * self.args.data_loss_weight + 
            chi_loss * self.args.chi_loss_weight +
            viol_loss * self.args.viol_loss_weight +
            center_loss * self.args.center_loss_weight
        )
        return gen_loss, total_gen_loss

    def loss_function(
            self,
            z_mu,
            z_logvar,
            y,
            y_recon,
            mask,
            beta,
            struct,
        ):

        # total of data loss and structural contraints
        gen_loss, total_gen_loss = self.gen_loss(y, y_recon, struct)

        # latent loss
        kld = torch.mean(
            -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        )
        if torch.isnan(kld):
            logger.info(z_mu[0])
            logger.info(z_logvar[0])
            raise RuntimeError("KLD is nan")

        # total loss
        if self.args.beta_control is None:
            loss = total_gen_loss + beta * kld / mask.sum().float()
        else:
            loss =  total_gen_loss+ self.args.beta_control * (beta - kld) ** 2 / mask.sum().float()

        return loss, gen_loss, kld

def save_checkpoint(model, optim, epoch, z_mu,z_logvar, out_weights, out_z):
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
    with open(out_z, "wb") as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)

def load_z(out_z):
    with open(out_z, "rb") as f:
        z_mu = pickle.load(f)
        z_logvar = pickle.load(f)
    return z_mu, z_logvar

class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        args = self.args
        if args.ind is not None:
            logger.info("Filtering image dataset with {}".format(args.ind))
            ind = pickle.load(open(args.ind, "rb"))
        else:
            ind = None

        imageDataset = dataset.ImageDataset(
            mrcfile=args.particles,
            lazy=args.lazy,
            norm=args.norm,
            invert_data=args.invert_data,
            ind=ind,
            window=args.window,
            datadir=args.datadir,
            window_r=args.window_r,
            max_threads=args.max_threads,
        )

        indices = np.arange(imageDataset.N)
        first_val_ind = int(self.args.train_val_ratio * imageDataset.N)
        indices_train = indices[:first_val_ind]
        indices_val = indices[first_val_ind:]

        self.train_data = dataset.DataSplits(imageDataset, indices_train)
        self.val_data  = dataset.DataSplits(imageDataset, indices_val)

        logger.info(f"{'='*50}")
        logger.info(f"Data Summary")
        logger.info(f"{'='*50}")
        total = self.train_data.N + self.val_data.N 
        logger.info("TOTAL PARTICLES       : %i"%(total))
        logger.info("training particles    : %i (%.2f %%)"%(self.train_data.N, 100* self.train_data.N /total))
        logger.info("validation particles  : %i (%.2f %%) "%(self.val_data.N, 100*  self.val_data.N /total ))
        logger.info(f"{'='*50}")

    def train_dataloader(self):
        args = self.args
        return dataset.make_dataloader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=args.num_workers,
            shuffler_size=self.args.shuffler_size,
            seed=self.args.shuffle_seed,
        )

    def val_dataloader(self):
        args = self.args
        return [
            dataset.make_dataloader( # This is the dataset used to validate 
                self.val_data,
                batch_size=self.args.batch_size,
                num_workers=args.num_workers,
                shuffler_size=self.args.shuffler_size,
                seed=self.args.shuffle_seed,
                shuffle=False
            ),
            dataset.make_dataloader( # This is not used in validation, I pass the rest of the data in order to record z latent projections at each validation step
                self.train_data,
                batch_size=self.args.batch_size,
                num_workers=args.num_workers,
                shuffler_size=self.args.shuffler_size,
                seed=self.args.shuffle_seed,
                shuffle=False
            ),
        ]
    
def main(args: argparse.Namespace) -> None:
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    torch.autograd.set_detect_anomaly(True)
    t1 = dt.now()
    if args.outdir is not None and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    ################################################################################""
    if args.overwrite:
        os.system("rm -rvf %s/*"%args.outdir )
    ################################################################################""


    logger.addHandler(logging.FileHandler(f"{args.outdir}/run.log"))
    logger.info(" ".join(sys.argv))
    logger.info(args)

    # set the random seed ------------------------------------------------------------------------------------------------------------------------
    pl.seed_everything(args.seed)

    # load dataset ------------------------------------------------------------------------------------------------------------------------
    logger.info(f"Loading dataset from {args.particles}")
    assert not ((args.domain != "real") and (args.encode_mode == "conv"))
    datamodule = LitDataModule(args)
    datamodule.setup()

    # load model ------------------------------------------------------------------------------------------------------------------------
    model = LitHetOnlyVAE(args, datamodule.train_data.D, datamodule.train_data.N + datamodule.val_data.N)

    if args.load:
        logger.info("Loading checkpoint from {}".format(args.load))
        checkpoint = torch.load(args.load)
        # filter unwanted keys
        exclude_prefixes = [
            "lattice",
            "decoder.embeddings",
        ]
        exclude_exact = {"decoder.rot_init", "decoder.trans_init"}

        filtered_state_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if not any(k.startswith(p) for p in exclude_prefixes)
            and k not in exclude_exact
        }

        # now load
        missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # optim = model.configure_optimizers()
        # optim.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Successfully restored states from {}".format(args.load))

    # save configuration
    out_config = "{}/config.yaml".format(args.outdir)
    save_config(args, datamodule.train_data.imageDataset, model.model.lattice, out_config)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=args.outdir,          # where to save
    #     filename="ckpt",  # filename pattern
    #     monitor="loss",              # what metric to track
    #     save_top_k=3,                    # save the 3 best models
    #     mode="min",                      # "min" for loss, "max" for accuracy, etc.
    #     save_last=False,                  # also save 'last.ckpt'
    #     verbose=True,
    # )

    cluster_environment = MPIEnvironment() if args.mpi_plugin else None

    n_devices = torch.cuda.device_count() if args.devices == "auto" else int(args.devices)

    if n_devices >1:
        strategy = DDPStrategy(find_unused_parameters=False,
                                cluster_environment=cluster_environment,
                                process_group_backend="nccl")
                                # process_group_backend="gloo")
    else:
        strategy="auto"


    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        strategy=strategy,
        devices=n_devices, 
        precision=args.precision,
        log_every_n_steps=10,
        callbacks=None,
        num_nodes = args.num_nodes ,
        logger=CSVLogger(args.outdir, name="", version=""),
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=datamodule)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)