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
from flexfold.core import ifft2_center


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
        "--mpi_plugin", action="store_true", help="TODO"
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
        "--devices", type=str, default="auto", help="TODO"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="TODO"
    )
    parser.add_argument(
        "--pair_stack", action="store_true", help="TODO"
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

def save_config(args, dataset, lattice, out_config):
    dataset_args = dict(
        particles=args.particles,
        norm=dataset.norm,
        invert_data=args.invert_data,
        ind=args.ind,
        keepreal=args.use_real,
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
        real_space = args.real_space,
        is_multimer = args.multimer,
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
            real_space=args.real_space,
            is_multimer = args.multimer,
            af_checkpoint_path =args.af_checkpoint_path,
        )

        self.val_z_mu = []
        self.val_z_logvar = []

        # for param in self.model.decoder.structure_module.parameters():
        #     param.requires_grad = False
    
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
        ind = batch[-1]

        B = ind.size(0)
        D = self.lattice.D

        # Pose
        rot, tran = self.posetracker.get_pose(ind)

        # Image
        if self.domain == "fourier":
            y = torch.view_as_real(batch[2])
            y = self.lattice.translate_ft(y.view(B, D*D, 2), tran.unsqueeze(1)).view(B, D, D, 2)
        else:
            y = batch[0]
            y = self.lattice.translate_ht(y.view(B, -1), tran.unsqueeze(1)).view(B, D, D)

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

        return y, rot, tran, c
    
    def write_debug(self, struct, mask, y, y_recon, global_it):
        print("Writing debug PNG at iteration %i"%global_it)
        D = self.lattice.D
        if B> 8 : 
            B=8
        fig, ax = plt.subplots(5, B, layout="constrained", figsize=(30,10))
        ax = np.array(ax).reshape(5, B) 
        masked_overlay = np.zeros((D* D, 4))
        masked_overlay[(mask == 0).detach().cpu().numpy()] = [0, 0, 0, 1]   # Black with full opacity
        masked_overlay[(mask == 1).detach().cpu().numpy()] = [0, 0, 0, 0]   # Fully transparent
        masked_overlay = masked_overlay.reshape((D,D,4))
        for i in range(B):
            y_ = torch.zeros((D*D), dtype=y.dtype, device=y.device)
            r_ = torch.zeros((D*D), dtype=y_recon.dtype, device=y_recon.device)
            if self.domain =="fourier":
                y_ = y[i, ..., 0] + 1j* y[i, ..., 1]
                r_[mask] = y_recon[i, ...,0] + 1j* y_recon[i, ...,1]
                r_ft = torch.zeros((D*D,2), dtype=y_recon.dtype, device=y_recon.device)
                r_ft[mask] = y_recon[i]

                y_real = fft.ifftn_center(torch.view_as_complex(y[i].reshape(D,D, 2))).real
                r_real = fft.ifftn_center(torch.view_as_complex(r_ft.reshape(D,D, 2))).real
            else:
                y_ = y[i]
                r_[mask] = y_recon[i]
                y_real = fft.ihtn_center(y[i].reshape(D,D))
                r_real = fft.ihtn_center(r_.reshape(D,D))
            y_ = y_.reshape(D,D)
            r_ = r_.reshape(D,D)
            y_real = y_real.reshape(D,D)
            r_real = r_real.reshape(D,D)

            y_*=gaussian_weight(D,0.1).to(y_.device)
            r_*=gaussian_weight(D,0.1).to(y_.device)

            y_ = y_.detach().cpu().numpy()
            r_ = r_.detach().cpu().numpy()
            y_real = y_real.detach().cpu().numpy()
            r_real = r_real.detach().cpu().numpy()

            #filter
            y_ = gaussian_filter(y_, sigma=3)
            r_ = gaussian_filter(r_, sigma=3)
            y_real = gaussian_filter(y_real, sigma=3)
            r_real = gaussian_filter(r_real, sigma=3)

            # Normalize
            def peak_normalize(img):
                m = np.max(np.abs(img), axis=(-2,-1), keepdims=True)
                return img / (m + 1e-8)
            
            def l2_normalize(img):
                norm = np.linalg.norm(img, axis=(-2, -1), keepdims=True)  
                return img / (norm + 1e-8)
            
            y_ =peak_normalize(y_)
            r_ =peak_normalize(r_)
            y_real =l2_normalize(y_real)
            r_real =l2_normalize(r_real)

            vmin= min(y_.min(), r_.min())
            vmax= min(y_.max(), r_.max())
            ax[0,i].imshow(y_, vmin=vmin, vmax=vmax, cmap="jet")
            ax[1,i].imshow(r_, vmin=vmin, vmax=vmax, cmap="jet")
            ax[2,i].imshow((np.abs(y_-r_)), cmap="jet")

            vmin= min(y_real.min(), r_real.min())
            vmax= min(y_real.max(), r_real.max())
            ax[3,i].imshow(y_real, vmin=vmin, vmax=vmax, cmap="Greys_r")
            ax[4,i].imshow(r_real, vmin=vmin, vmax=vmax, cmap="Greys_r")

            ax[0,i].imshow(masked_overlay)
            ax[1,i].imshow(masked_overlay)
            ax[2,i].imshow(masked_overlay)
            ax[0,i].axis('off')
            ax[1,i].axis('off')
            ax[2,i].axis('off')
            ax[3,i].axis('off')
            ax[4,i].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(self.args.outdir + "/debug_%s.png"%str(global_it).zfill(5))
        plt.close(fig)
        output_single_pdb(
            all_atom_positions= (struct["final_atom_positions"] @ self.model.decoder.rot_init + self.model.decoder.trans_init[..., None, :]).detach().cpu().numpy()[-1],
            all_atom_mask=struct["all_atom_mask"].detach().cpu().numpy()[-1],
            aatype=struct["aatype"].detach().cpu().numpy()[-1],
            residue_index=struct["residue_index"].detach().cpu().numpy()[-1],
            chain_index=struct["asym_id"].detach().cpu().numpy()[-1],
            file= self.args.outdir + "/debug_%s.pdb"%str(global_it).zfill(5)
        )
    def run_encoder(self, y, c=None):
        if self.domain == "fourier":
            y = (y[...,0] - y[...,1]) 
        input_ = (y,)
        if c is not None:
            input_ = (x * c.sign() for x in input_)  # phase flip by the ctf

        z_mu, z_logvar = self.model.encode(*input_)

        return z_mu, z_logvar 

    def run_decoder(self, z, rot, c):
        B = z.size(0)
        # decode
        mask = self.lattice.get_circular_mask(self.lattice.D // 2)  # restrict to circular mask

        if isinstance(self.model.decoder, AFDecoderReal):
            y_recon, struct, y_recon_real = self.model(rot, z)
            y_recon =  y_recon.view(B, -1)[..., mask]
        elif isinstance(self.model.decoder, AFDecoder):
            y_recon, struct = self.model(self.lattice.coords[mask] / self.lattice.extent / 2 @ rot, z)
        else:
            y_recon = self.model(self.lattice.coords[mask] / self.lattice.extent / 2 @ rot, z)

        y_recon = y_recon.view(B, -1)
        y_recon *= c.view(B, -1)[:, mask]

        if self.domain =="fourier":
            y_recon = torch.view_as_real(y_recon).to(torch.float32)

        return y_recon, mask, struct,# y_tmp, y_recon_real


    def training_step(self, batch, batch_idx):
        
        # Preprocessing
        y, rot, _, c = self.prepare_batch(batch)

        B = batch[-1].size(0)
        global_it = self.Nparticles * self.current_epoch +  batch_idx * B
        beta = self.beta_schedule( global_it)

        # Encdoer
        z_mu, z_logvar = self.run_encoder(y, c)

        # Reparametrize latent space
        z = self.model.reparameterize(z_mu, z_logvar)

        # Decoder
        y_recon, mask, struct = self.run_decoder(z,rot, c)

        # Write debug
        if self.args.debug and self.trainer.is_global_zero and ((global_it)%100 == 0  or batch_idx ==0):
            self.write_debug(struct, mask, y, y_recon, global_it)

        # Computing loss
        loss, gen_loss, kld = self.loss_function(
            z_mu,
            z_logvar,
            y,
            y_recon,
            mask,
            beta,
            struct=struct
        )

        # Logging
        self.log("loss", loss, prog_bar=True, sync_dist=True, on_epoch=True, on_step=True)
        self.log("kld", kld, prog_bar=False, sync_dist=True, on_epoch=True, on_step=True)
        for k,v in gen_loss.items():
            self.log(k, v, prog_bar=False, sync_dist=True, on_epoch=True, on_step=True)

        return loss


    def validation_step(self, batch, batch_idx):
        y, _, _, c = self.prepare_batch(batch)
        z_mu, z_logvar = self.run_encoder(y, c)
        self.val_z_mu.append(z_mu)
        self.val_z_logvar.append(z_logvar)
    
    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero: 
            z_mu_all = torch.cat(self.val_z_mu).to(torch.float32).detach().cpu().numpy()
            z_logvar_all = torch.cat(self.val_z_logvar).to(torch.float32).detach().cpu().numpy()


            out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.current_epoch)
            out_z = "{}/z.{}.pkl".format(self.args.outdir, self.current_epoch)

            
            save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu_all, z_logvar_all, out_weights, out_z)
            self.val_z_mu.clear()
            self.val_z_logvar.clear() 

    # def on_validation_epoch_end(self):

    #     z_mu_local = torch.cat(self.val_z_mu, dim=0).detach().cpu().numpy()
    #     z_logvar_local = torch.cat(self.val_z_logvar, dim=0).detach().cpu().numpy()

    #     world_size = torch.distributed.get_world_size()
    #     gathered_mu, gathered_logvar = [None]*world_size, [None]*world_size

    #     torch.distributed.all_gather_object(gathered_mu, z_mu_local)
    #     torch.distributed.all_gather_object(gathered_logvar, z_logvar_local)

    #     if torch.distributed.get_rank() == 0:
    #         z_mu_all = np.concatenate(gathered_mu, axis=0)
    #         z_logvar_all = np.concatenate(gathered_logvar, axis=0)
    #         out_weights = "{}/weights.{}.pkl".format(self.args.outdir, self.current_epoch)
    #         out_z = "{}/z.{}.pkl".format(self.args.outdir, self.current_epoch)
            
    #         save_checkpoint(self.model, self.optimizers(), self.current_epoch, z_mu_all, z_logvar_all, out_weights, out_z)
    

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

        # Reconstruction loss
        D =  y.shape[-2]
        B = y.size(0)

        if self.domain =="fourier":
            r_ = torch.zeros((B, D*D, 2), dtype=y_recon.dtype, device=y_recon.device)
            y = y.view(B, D*D, 2)
            r_[:, mask] = y_recon[:]
            y_real = ifft2_center(torch.view_as_complex(y.reshape(B, D,D, 2))).real
            r_real = ifft2_center(torch.view_as_complex(r_.reshape(B, D,D, 2))).real
        else:
            r_ = torch.zeros((B, D*D), dtype=y_recon.dtype, device=y_recon.device)
            y = y.view(B, D*D)
            r_[:, mask] = y_recon[:]
            y_real = fft.iht2_center(y.reshape(B,D,D))
            r_real = fft.iht2_center(r_.reshape(B,D,D))

        cc = get_cc(y_real, r_real)
        data_loss = torch.mean(1-cc)

        # if self.domain=="fourier":
        #     y = y.view(B, D*D, 2)[:, mask]
        #     cc = fourier_corr(torch.view_as_complex(y), torch.view_as_complex(y_recon))
        #     cc = torch.mean(cc)


        # else:
        #     y = y.view(B, -1)[:, mask]
        #     cc = get_cc(y_recon, y)
        #     cc = torch.mean(cc)

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

        # optionally split data if needed; skip if you only have train
        self.train_data = dataset.ImageDataset(
            mrcfile=args.particles,
            lazy=args.lazy,
            norm=args.norm,
            invert_data=args.invert_data,
            ind=ind,
            keepreal=args.use_real,
            window=args.window,
            datadir=args.datadir,
            window_r=args.window_r,
            max_threads=args.max_threads,
        )


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
        return dataset.make_dataloader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=args.num_workers,
            shuffler_size=self.args.shuffler_size,
            seed=self.args.shuffle_seed,
            shuffle=False
        )
    
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
    args.use_real = args.encode_mode == "conv"  # Must be False
    datamodule = LitDataModule(args)
    datamodule.setup()

    # load model ------------------------------------------------------------------------------------------------------------------------
    model = LitHetOnlyVAE(args, datamodule.train_data.D, datamodule.train_data.N)

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
    save_config(args, datamodule.train_data, model.model.lattice, out_config)

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

    strategy = DDPStrategy(find_unused_parameters=False,
                            cluster_environment=cluster_environment,
                            process_group_backend="nccl")
                            # process_group_backend="gloo")


    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        strategy=strategy,
        devices=args.devices, 
        precision=args.precision,
        log_every_n_steps=10,
        callbacks=None,
        num_nodes = args.num_nodes ,
        logger=CSVLogger(args.outdir, name="", version=""),
        max_steps=10
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)