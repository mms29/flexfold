"""Pytorch models"""

from typing import Optional, Tuple, Type, Sequence, Any
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.parallel import DataParallel
from cryodrgn import fft, lie_tools, utils
import cryodrgn.config

from openfold.utils.tensor_utils import tensor_tree_map
from openfold.model.structure_module import StructureModule
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    atom14_to_atom37,
)
from functools import reduce
import operator


from cryodrgn.mrcfile import write_mrc, parse_mrc
from openfold.config import model_config
import openfold.np.residue_constants as rc
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)
import os

from flexfold.lattice import Lattice
from flexfold.core import (img_ft_lattice, img_ht_lattice, img_real, get_pixel_mask, 
                            img_real_mask, vol_real_mask, get_voxel_mask, register_crd_to_vol)

from openfold.utils.import_weights import convert_deprecated_v1_keys
from openfold.data import data_transforms
from openfold.model.template import TemplatePairStack
from openfold.model.primitives import Linear
Norm = Sequence[Any]  # mean, std


def unparallelize(model: nn.Module) -> nn.Module:
    if isinstance(model, DataParallelDecoder):
        return model.dp.module
    if isinstance(model, DataParallel):
        return model.module
    return model


class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(
        self,
        lattice: Lattice,
        qlayers: int,
        qdim: int,
        players: int,
        pdim: int,
        in_dim: int,
        zdim: int = 1,
        encode_mode: str = "resid",
        enc_mask=None,
        enc_type="linear_lowf",
        enc_dim=None,
        domain="fourier",
        activation=nn.ReLU,
        feat_sigma: Optional[float] = None,
        tilt_params={},


        # added paramrs
        af_decoder = False,
        initial_pose_path = None,
        embedding_path="./output/cryofold/AK_embeddings.pt",
        af_checkpoint_path = "../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt",
        sigma = 2.5,
        pixel_size = 2.2,
        quality_ratio = 5,
        real_space = True,
        all_atom=True,
        pair_stack = False,
        is_multimer=False,
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == "conv":
            self.encoder = ConvEncoder(qdim, zdim * 2)
        elif encode_mode == "resid":
            self.encoder = ResidLinearMLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,  # nlayers  # hidden_dim  # out_dim
            )
        elif encode_mode == "mlp":
            self.encoder = MLP(
                in_dim, qlayers, qdim, zdim * 2, activation  # hidden_dim  # out_dim
            )  # in_dim -> hidden_dim
        elif encode_mode == "tilt":
            self.encoder = TiltEncoder(
                in_dim,
                qlayers,
                qdim,
                tilt_params["t_emb_dim"],  # embedding dim
                tilt_params["ntilts"],  # number of encoded tilts
                tilt_params["tlayers"],
                tilt_params["tdim"],
                zdim * 2,  # outdim
                activation,
            )
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        self.encode_mode = encode_mode

        if af_decoder :
            self.decoder = get_afdecoder(
                zdim= zdim,
                lattice_size =lattice.D,
                layers= players,
                hidden_dim = pdim,
                activation = activation,
                embedding_path = embedding_path,
                af_checkpoint_path = af_checkpoint_path,
                initial_pose_path = initial_pose_path,
                sigma = sigma,
                pixel_size = pixel_size,
                real_space=real_space,
                quality_ratio=quality_ratio,
                all_atom = all_atom,
                pair_stack = pair_stack,
                domain = domain,
                is_multimer=is_multimer,
            )
        else:
            self.decoder = get_decoder(
                3 + zdim,
                lattice.D,
                players,
                pdim,
                domain,
                enc_type,
                enc_dim,
                activation,
                feat_sigma,
            )

    @classmethod
    def load(cls, config, weights=None, device=None):
        """Instantiate a model from a config.yaml

        Inputs:
            config (str, dict): Path to config.yaml or loaded config.yaml
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        """
        cfg = cryodrgn.config.load(config)

        c = cfg["lattice_args"]
        print("hello")
        lat = Lattice(c["D"], extent=c["extent"])
        c = cfg["model_args"]
        if c["enc_mask"] > 0:
            enc_mask = lat.get_circular_mask(c["enc_mask"])
            in_dim = int(enc_mask.sum())
        else:
            assert c["enc_mask"] == -1
            enc_mask = None
            in_dim = lat.D**2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
        model = HetOnlyVAE(
            lat,
            c["qlayers"],
            c["qdim"],
            c["players"],
            c["pdim"],
            in_dim,
            c["zdim"],
            encode_mode=c["encode_mode"],
            enc_mask=enc_mask,
            enc_type=c["pe_type"],
            enc_dim=c["pe_dim"],
            domain=c["domain"],
            activation=activation,
            feat_sigma=c["feat_sigma"],
            tilt_params=c.get("tilt_params", {}),

            af_decoder = c["af_decoder"],
            initial_pose_path = c["initial_pose_path"],
            embedding_path=c["embedding_path"],
            af_checkpoint_path = c["af_checkpoint_path"],
            sigma = c["sigma"],
            pixel_size = c["pixel_size"],
            quality_ratio = c["quality_ratio"],
            real_space = c["real_space"],
            all_atom = c["all_atom"],
            pair_stack = c["pair_stack"] if "pair_stack" in c else False,
            is_multimer = c["is_multimer"]
        )
        if weights is not None:
            print(weights)
            ckpt = torch.load(weights)
            print(ckpt)
            model.load_state_dict(ckpt["model_state_dict"])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, *img) -> Tuple[Tensor, Tensor]:
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z[:, : self.zdim], z[:, self.zdim :]

    def cat_z(self, coords, z) -> Tensor:
        """
        coords: Bx...x3
        z: Bxzdim
        """
        assert coords.size(0) == z.size(0), (coords.shape, z.shape)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z=None) -> torch.Tensor:
        """
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        """
        decoder = self.decoder
        assert isinstance(decoder, nn.Module)

        if isinstance(decoder, AFDecoder):
            retval = decoder(coords, z)
        elif isinstance(decoder, AFDecoderReal):
            # coords are actually rot !!
            retval = decoder(coords, z)
        else:
            retval = decoder(self.cat_z(coords, z) if z is not None else coords)
        return retval

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


def load_decoder(config, weights=None, device=None):
    """
    Instantiate a decoder model from a config.yaml

    Inputs:
        config (str, dict): Path to config.yaml or loaded config.yaml
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    """
    cfg = cryodrgn.config.load(config)
    c = cfg["model_args"]
    D = cfg["lattice_args"]["D"]
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
    model = get_decoder(
        3,
        D,
        c["layers"],
        c["dim"],
        c["domain"],
        c["pe_type"],
        c["pe_dim"],
        activation,
        c["feat_sigma"],
    )
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model.to(device)
    return model


class Decoder(nn.Module):
    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume
        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        raise NotImplementedError

    def get_voxel_decoder(self) -> Optional["Decoder"]:
        return None


class DataParallelDecoder(Decoder):
    def __init__(self, decoder: Decoder):
        super(DataParallelDecoder, self).__init__()
        self.dp = torch.nn.parallel.DataParallel(decoder)

    def eval_volume(self, *args, **kwargs):
        module = self.dp.module
        assert isinstance(module, Decoder)
        return module.eval_volume(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.dp.module.forward(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.dp.module.state_dict(*args, **kwargs)


class PositionalDecoder(Decoder):
    def __init__(
        self,
        in_dim,
        D,
        nlayers,
        hidden_dim,
        activation,
        enc_type="linear_lowf",
        enc_dim=None,
        feat_sigma: Optional[float] = None,
    ):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)

        if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma
            rand_freqs = (
                torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            )
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None

    def positional_encoding_geom(self, coords):
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == "geom_ft":
            freqs = (
                self.DD * np.pi * (2.0 / self.DD) ** (freqs / (self.enc_dim - 1))
            )  # option 1: 2/D to 1
        elif self.enc_type == "geom_full":
            freqs = (
                self.DD
                * np.pi
                * (1.0 / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
            )  # option 2: 2/D to 2pi
        elif self.enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 3: 2/D*2pi to 2pi
        elif self.enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 4: 2/D*2pi to 1
        elif self.enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = coords[..., None, 0:3] * freqs  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        """Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2"""
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, coords: Tensor) -> Tensor:
        """Input should be coordinates from [-.5,.5]"""
        assert (coords[..., 0:3].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        assert extent <= 0.5
        zdim = 0
        z = torch.tensor([])
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32, device=coords.device)

        vol_f = torch.zeros((D, D, D), dtype=torch.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(
            np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)
        ):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D, D)
            vol_f[i] = y

        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.ihtn_center(vol_f[0:-1, 0:-1, 0:-1])
        return vol


class FTPositionalDecoder(Decoder):
    def __init__(
        self,
        in_dim: int,
        D: int,
        nlayers: int,
        hidden_dim: int,
        activation: Type,
        enc_type: str = "linear_lowf",
        enc_dim: Optional[int] = None,
        feat_sigma: Optional[float] = None,
    ):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

        if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma
            rand_freqs = (
                torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            )
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = Parameter(rand_freqs, requires_grad=False)
        else:
            self.rand_feats = None

    def positional_encoding_geom(self, coords: Tensor) -> Tensor:
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        if self.enc_type == "gaussian":
            return self.random_fourier_encoding(coords)
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == "geom_ft":
            freqs = (
                self.DD * np.pi * (2.0 / self.DD) ** (freqs / (self.enc_dim - 1))
            )  # option 1: 2/D to 1
        elif self.enc_type == "geom_full":
            freqs = (
                self.DD
                * np.pi
                * (1.0 / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
            )  # option 2: 2/D to 2pi
        elif self.enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 3: 2/D*2pi to 2pi
        elif self.enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 4: 2/D*2pi to 1
        elif self.enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = coords[..., None, 0:3] * freqs  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords: Tensor) -> Tensor:
        """Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2"""
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice: Tensor) -> Tensor:
        """
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        """
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2] // 2  # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c  # include the origin
        assert abs(lattice[..., 0:3].mean()) < 1e-4, "{} != 0.0".format(
            lattice[..., 0:3].mean()
        )
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top_half = self.decode(lattice[..., 0:cc, :])
        image[..., 0:cc] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., cc:] = (top_half[..., 0] + top_half[..., 1])[
            ..., np.arange(c - 1, -1, -1)
        ]
        return image

    def decode(self, lattice: Tensor):
        """Return FT transform"""
        assert (lattice[..., 0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[..., 2] > 0.0
        new_lattice = lattice.clone()
        # negate lattice coordinates where z > 0
        new_lattice[..., 0:3][w] *= -1
        result = self.decoder(self.positional_encoding_geom(new_lattice))
        # replace with complex conjugate to get correct values for original lattice positions
        result[..., 1][w] *= -1
        return result

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        assert extent <= 0.5
        zdim = 0
        z = torch.tensor([])
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32, device=coords.device)

        vol_f = torch.zeros((D, D, D), dtype=torch.float32, device=coords.device)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(
            np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)
        ):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            keep = x.pow(2).sum(dim=1) <= extent**2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros(D**2, device=coords.device)
                slice_[keep] = y
                slice_ = slice_.view(D, D)
            vol_f[i] = slice_
        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.ihtn_center(
            vol_f[:-1, :-1, :-1]
        )  # remove last +k freq for inverse FFT
        return vol


class FTSliceDecoder(Decoder):
    """
    Evaluate a central slice out of a 3D FT of a model, returns representation in
    Hartley reciprocal space

    Exploits the symmetry of the FT where F*(x,y) = F(-x,-y) and only
    evaluates half of the lattice. The decoder is f(x,y,z) => real, imag
    """

    def __init__(self, in_dim: int, D: int, nlayers: int, hidden_dim: int, activation):
        """D: image width or height"""
        super(FTSliceDecoder, self).__init__()
        self.decoder = ResidLinearMLP(in_dim, nlayers, hidden_dim, 2, activation)
        D2 = int(D / 2)

        # various pixel indices to keep track of for forward_even
        self.center = D2 * D + D2
        self.extra = np.arange(
            (D2 + 1) * D, D**2, D
        )  # bottom-left column without conjugate pair
        # evalute the top half of the image up through the center pixel
        # and extra bottom-left column (todo: just evaluate a D-1 x D-1 image so
        # we don't have to worry about this)
        self.all_eval = np.concatenate((np.arange(self.center + 1), self.extra))

        # pixel indices for the top half of the image up to (but not incl)
        # the center pixel and excluding the top row and left-most column
        i, j = np.meshgrid(np.arange(1, D), np.arange(1, D2 + 1))
        self.top = (j * D + i).ravel()[:-D2]

        # pixel indices for bottom half of the image after the center pixel
        # excluding left-most column and given in reverse order
        i, j = np.meshgrid(np.arange(1, D), np.arange(D2, D))
        self.bottom_rev = (j * D + i).ravel()[D2:][::-1].copy()

        self.D = D
        self.D2 = D2

    def forward(self, lattice):
        """
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        """
        assert lattice.shape[-2] % 2 == 1
        c = lattice.shape[-2] // 2  # center pixel
        assert lattice[..., c, 0:3].sum() == 0.0, "{} != 0.0".format(
            lattice[..., c, 0:3].sum()
        )
        assert abs(lattice[..., 0:3].mean()) < 1e-4, "{} != 0.0".format(
            lattice[..., 0:3].mean()
        )
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top_half = self.decode(lattice[..., 0 : c + 1, :])
        image[..., 0 : c + 1] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., c + 1 :] = (top_half[..., 0] + top_half[..., 1])[
            ..., np.arange(c - 1, -1, -1)
        ]
        return image

    def forward_even(self, lattice):
        """Extra bookkeeping with extra row/column for an even sized DFT"""
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top_half = self.decode(lattice[..., self.all_eval, :])
        image[..., self.all_eval] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., self.bottom_rev] = (
            top_half[..., self.top, 0] + top_half[..., self.top, 1]
        )
        return image

    def decode(self, lattice):
        """Return FT transform"""
        # convention: only evalute the -z points
        w = lattice[..., 2] > 0.0
        new_lattice = lattice.clone()
        # negate lattice coordinates where z > 0

        new_lattice[..., 0:3][w] *= -1
        result = self.decoder(new_lattice)
        # replace with complex conjugate to get correct values for original lattice positions
        result[..., 1][w] *= -1
        return result

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32, device=coords.device)
        else:
            z = None

        vol_f = torch.zeros((D, D, D), dtype=torch.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(
            np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)
        ):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            if zval is not None:
                assert z is not None
                x = torch.cat((x, z), dim=-1)
            with torch.no_grad():
                y = self.decode(x)
                y = y[..., 0] - y[..., 1]
                y = y.view(D, D).cpu()
            vol_f[i] = y
        vol_f = vol_f * norm[1] + norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(
            vol_f[:-1, :-1, :-1]
        )  # remove last +k freq for inverse FFT
        return vol


def get_decoder(
    in_dim: int,
    D: int,
    layers: int,
    dim: int,
    domain: str,
    enc_type: str,
    enc_dim: Optional[int] = None,
    activation: Type = nn.ReLU,
    feat_sigma: Optional[float] = None,
) -> Decoder:

    if enc_type == "none":
        if domain == "hartley":
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
        else:
            model = FTSliceDecoder(in_dim, D, layers, dim, activation)
    else:
        model_t = PositionalDecoder if domain == "hartley" else FTPositionalDecoder
        model = model_t(
            in_dim,
            D,
            layers,
            dim,
            activation,
            enc_type=enc_type,
            enc_dim=enc_dim,
            feat_sigma=feat_sigma,
        )
    return model


class VAE(nn.Module):
    def __init__(
        self,
        lattice,
        qlayers: int,
        qdim: int,
        players: int,
        pdim: int,
        encode_mode: str = "mlp",
        no_trans: bool = False,
        enc_mask: Optional[Tensor] = None,
    ):
        super(VAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        if enc_mask is not None:
            self.in_dim = (
                lattice.D * lattice.D if enc_mask is None else int(enc_mask.sum())
            )
        self.enc_mask = enc_mask
        assert qlayers > 2
        if encode_mode == "conv":
            self.encoder = ConvEncoder(qdim, qdim)
        elif encode_mode == "resid":
            self.encoder = ResidLinearMLP(
                self.in_dim,
                qlayers - 2,  # -2 bc we add 2 more layers in the homeomorphic encoer
                qdim,  # hidden_dim
                qdim,  # out_dim
                nn.ReLU,
            )  # in_dim -> hidden_dim
        elif encode_mode == "mlp":
            self.encoder = MLP(
                self.in_dim, qlayers - 2, qdim, qdim, nn.ReLU  # hidden_dim  # out_dim
            )  # in_dim -> hidden_dim
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        # predict rotation and translation in two completely separate NNs
        # self.so3_encoder = SO3reparameterize(qdim) # hidden_dim -> SO(3) latent variable
        # self.trans_encoder = ResidLinearMLP(nx*ny, 5, qdim, 4, nn.ReLU)

        # or predict rotation/translations from intermediate encoding
        self.so3_encoder = SO3reparameterize(
            qdim, 1, qdim
        )  # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(qdim, 1, qdim, 4, nn.ReLU)

        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        self.no_trans = no_trans

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, img) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """img: BxDxD"""
        img = img.view(img.size(0), -1)
        if self.enc_mask is not None:
            img = img[:, self.enc_mask]
        enc = nn.ReLU()(self.encoder(img))
        z_mu, z_std = self.so3_encoder(enc)
        if self.no_trans:
            tmu, tlogvar = (None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:, :2], z[:, 2:]
        return z_mu, z_std, tmu, tlogvar

    def eval_volume(self, norm) -> Tensor:
        return self.decoder.eval_volume(
            self.lattice.coords, self.D, self.lattice.extent, norm
        )

    def decode(self, rot):
        # transform lattice by rot.T
        x = self.lattice.coords @ rot  # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)
        return y_hat

    def forward(self, img: Tensor):
        z_mu, z_std, tmu, tlogvar = self.encode(img)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        # transform lattice by rot and predict image
        y_hat = self.decode(rot)
        if not self.no_trans:
            # translate image by t
            assert tmu is not None and tlogvar is not None
            B = img.size(0)
            t = self.reparameterize(tmu, tlogvar)
            t = t.unsqueeze(1)  # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B, -1), t)
            img = img.view(B, self.D, self.D)
        return y_hat, img, z_mu, z_std, w_eps, tmu, tlogvar


class TiltVAE(nn.Module):
    def __init__(
        self, lattice, tilt, qlayers, qdim, players, pdim, no_trans=False, enc_mask=None
    ):
        super(TiltVAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D * lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 3
        self.encoder = ResidLinearMLP(self.in_dim, qlayers - 3, qdim, qdim, nn.ReLU)
        self.so3_encoder = SO3reparameterize(
            2 * qdim, 3, qdim
        )  # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(2 * qdim, 2, qdim, 4, nn.ReLU)
        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        assert tilt.shape == (3, 3), "Rotation matrix input required"
        self.tilt = torch.tensor(tilt)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def eval_volume(self, norm) -> Tensor:
        return self.decoder.eval_volume(
            self.lattice.coords, self.D, self.lattice.extent, norm
        )

    def encode(self, img, img_tilt):
        img = img.view(img.size(0), -1)
        img_tilt = img_tilt.view(img_tilt.size(0), -1)
        if self.enc_mask is not None:
            img = img[:, self.enc_mask]
            img_tilt = img_tilt[:, self.enc_mask]
        enc1 = self.encoder(img)
        enc2 = self.encoder(img_tilt)
        enc = torch.cat((enc1, enc2), -1)  # then nn.ReLU?
        z_mu, z_std = self.so3_encoder(enc)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        if self.no_trans:
            tmu, tlogvar, t = (None, None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:, :2], z[:, 2:]
            t = self.reparameterize(tmu, tlogvar)
        return z_mu, z_std, w_eps, rot, tmu, tlogvar, t

    def forward(self, img, img_tilt):
        B = img.size(0)
        z_mu, z_std, w_eps, rot, tmu, tlogvar, t = self.encode(img, img_tilt)
        if not self.no_trans:
            assert t is not None
            t = t.unsqueeze(1)  # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B, -1), -t)
            img_tilt = self.lattice.translate_ht(img_tilt.view(B, -1), -t)
            img = img.view(B, self.D, self.D)
            img_tilt = img_tilt.view(B, self.D, self.D)

        # rotate lattice by rot.T
        x = self.lattice.coords @ rot  # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)

        # tilt series pair
        x = self.lattice.coords @ self.tilt @ rot
        y_hat2 = self.decoder(x)
        y_hat2 = y_hat2.view(-1, self.D, self.D)
        return y_hat, y_hat2, img, img_tilt, z_mu, z_std, w_eps, tmu, tlogvar


# fixme: this is half-deprecated (not used in TiltVAE, but still used in tilt BNB)
class TiltEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        nlayers,
        hidden_dim,
        out_dim,
        ntilts,
        nlayers2,
        hidden_dim2,
        out_dim2,
        activation,
    ):
        super(TiltEncoder, self).__init__()
        self.encoder1 = ResidLinearMLP(in_dim, nlayers, hidden_dim, out_dim, activation)
        self.encoder2 = ResidLinearMLP(
            out_dim * ntilts, nlayers2, hidden_dim2, out_dim2, activation
        )
        self.in_dim = in_dim
        self.in_dim2 = out_dim * ntilts

    def forward(self, x):
        x = self.encoder1(x)
        z = self.encoder2(x.view(-1, self.in_dim2))
        return z


class ResidLinearMLP(Decoder):
    def __init__(
        self,
        in_dim: int,
        nlayers: int,
        hidden_dim: int,
        out_dim: int,
        activation: Type,
    ):
        super(ResidLinearMLP, self).__init__()
        layers = [
            (
                ResidLinear(in_dim, hidden_dim)
                if in_dim == hidden_dim
                else MyLinear(in_dim, hidden_dim)
            ),
            activation(),
        ]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(
            ResidLinear(hidden_dim, out_dim)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret

    def eval_volume(
        self, coords: Tensor, D: int, extent: float, norm: Norm, zval=None
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32, device=coords.device)
            z += torch.tensor(zval, dtype=torch.float32, device=coords.device)

        vol_f = torch.zeros((D, D, D), dtype=torch.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(
            np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)
        ):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            if zval is not None:
                x = torch.cat((x, zval), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D, D).cpu()
            vol_f[i] = y
        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.ihtn_center(
            vol_f[0:-1, 0:-1, 0:-1]
        )  # remove last +k freq for inverse FFT
        return vol


def half_linear(input, weight, bias):
    # print('half', input.shape, weight.shape)
    return F.linear(input, weight.half(), bias.half())


def single_linear(input, weight, bias):
    # print('single', input.shape, weight.shape)
    # assert input.shape[0] < 10000

    return F.linear(input, weight, bias)


class MyLinear(nn.Linear):
    def forward(self, input):
        if input.dtype == torch.half:
            return half_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight.half(), self.bias.half())
        else:
            return single_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight, self.bias)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = MyLinear(nin, nout)
        # self.linear = nn.utils.weight_norm(MyLinear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nlayers: int,
        hidden_dim: int,
        out_dim: int,
        activation: Type,
    ):
        super(MLP, self).__init__()
        layers = [MyLinear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(MyLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(MyLinear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


class SO3reparameterize(nn.Module):
    """Reparameterize R^N encoder output to SO(3) latent variable"""

    def __init__(self, input_dims, nlayers: int, hidden_dim: int):
        super().__init__()
        if nlayers is not None:
            self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
        else:
            self.main = MyLinear(input_dims, 9)

        # start with big outputs
        # self.s2s2map.weight.data.uniform_(-5,5)
        # self.s2s2map.bias.data.uniform_(-5,5)

    def sampleSO3(
        self, z_mu: torch.Tensor, z_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterize SO(3) latent variable
        # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
        # See section 2.5 of http://ethaneade.com/lie.pdf
        """
        # resampling trick
        if not self.training:
            return z_mu, z_std
        eps = torch.randn_like(z_std)
        w_eps = eps * z_std
        rot_eps = lie_tools.expmap(w_eps)
        # z_mu = lie_tools.quaternions_to_SO3(z_mu)
        rot_sampled = z_mu @ rot_eps
        return rot_sampled, w_eps

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.main(x)
        z1 = z[:, :3].double()
        z2 = z[:, 3:6].double()
        z_mu = lie_tools.s2s2_to_SO3(z1, z2).float()
        logvar = z[:, 6:]
        z_std = torch.exp(0.5 * logvar)  # or could do softplus
        return z_mu, z_std

def get_afdecoder(
        zdim: int,
        lattice_size: int,
        layers: int,
        hidden_dim: int,
        activation: str,
        sigma: float,
        pixel_size: float,
        embedding_path: str,
        af_checkpoint_path: str,
        initial_pose_path: str,
        real_space: bool = False,
        quality_ratio: float = 5.0,
        all_atom: bool = True,
        pair_stack: bool =False,
        is_multimer: bool =False,
        domain: str = "hartley"
):
    embeddings = torch.load(embedding_path,map_location='cuda:0')

    initial_pose = torch.load(initial_pose_path)

    config = model_config(
        "initial_training", 
        train=True, 
        low_prec=False,
    ) 

    afdecoder_args = {
        "config": config,
        "embeddings":{k: torch.as_tensor(v) for k, v in embeddings.items()},
        "rot_init":torch.as_tensor(initial_pose["R"], dtype=torch.float32),
        "trans_init":torch.as_tensor(initial_pose["T"], dtype=torch.float32),
        "pixel_size":pixel_size,
        "sigma":sigma,
        "zdim":zdim,
        "lattice_size":lattice_size,
        "layers":layers,
        "hidden_dim":hidden_dim,
        "activation":activation,
        "quality_ratio":quality_ratio,
        "all_atom" : all_atom,
        "pair_stack" : pair_stack,
        "domain" : domain,
        "is_multimer":is_multimer,
    }

    if real_space:
        model = AFDecoderReal(**afdecoder_args)
    else:
        model = AFDecoder(**afdecoder_args)

    model = resume_ckpt(
        af_checkpoint_path, 
        model, 
        config
    )
    return model

class BufferDict(nn.Module):
    def __init__(self, buffer_dict: dict):
        super().__init__()
        for k, v in buffer_dict.items():
            self.register_buffer(k, v)

    def keys(self):
        return dict(self.named_buffers()).keys()

    def items(self):
        return dict(self.named_buffers()).items()

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        return hasattr(self, k)

    def __len__(self):
        return len(dict(self.named_buffers()))

class AFDecoder(torch.nn.Module):
    def __init__(self, 
                 config, 
                 embeddings, 
                 rot_init, 
                 trans_init, 
                 pixel_size, 
                 sigma,
                zdim,
                lattice_size,
                layers,
                hidden_dim,
                activation, 
                quality_ratio, 
                all_atom,
                pair_stack,
                domain,
                is_multimer               
        ):
        super(AFDecoder, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.loss_config = config.loss

        self.structure_module = StructureModule(
            is_multimer=is_multimer,
            **self.config["structure_module"],
        )

        self.pixel_size=pixel_size
        self.sigma = sigma
        self.lattice_size=lattice_size
        self.all_atom = all_atom
        self.pair_stack = pair_stack
        self.domain = domain
        self.is_multimer=is_multimer

        self.register_buffer("rot_init", rot_init)
        self.register_buffer("trans_init", trans_init)

        NUM_RES = "num residues placeholder"

        # self.embeddings = embeddings
        embeddings_keys = {
            "aatype": [NUM_RES],
            "seq_mask": [NUM_RES],
            "pair": [NUM_RES, NUM_RES, None],
            "single": [NUM_RES, None],
            "final_atom_mask":[NUM_RES, None],
            "final_atom_positions":[NUM_RES,None, None],
            "residx_atom37_to_atom14":[NUM_RES, None],
            "residx_atom14_to_atom37":[NUM_RES, None],
            "atom37_atom_exists":[NUM_RES, None],
            "atom14_atom_exists":[NUM_RES, None],
            "residue_index":[NUM_RES],
        }
        
        embeddings = {k: v for k, v in embeddings.items() if k in embeddings_keys.keys()}

        def make_gt(feats):
            feats["all_atom_positions"] = feats["final_atom_positions"] 
            feats["all_atom_mask"] = feats["final_atom_mask"] 
            return feats
    
        fc = [
            # data_transforms.make_fixed_size(embeddings_keys,0,0,500,0),
            make_gt,
            data_transforms.atom37_to_torsion_angles(""),
            data_transforms.get_chi_angles,
        ]
        for f in fc:
            embeddings = f(embeddings)

        self.embeddings = BufferDict(embeddings)

        self.res_size = self.embeddings["pair"].shape[-2]
        self.outdim = self.embeddings["pair"].shape[-1]


        self.zdim = zdim

        if self.pair_stack : 

            self.decoder_ = TemplatePairStack(
                    c_t=hidden_dim+zdim,
                    c_hidden_tri_att= 32,
                    c_hidden_tri_mul= 64,
                    no_blocks= layers,
                    no_heads= 4,
                    pair_transition_n= layers,
                    dropout_rate= 0.15,
                    tri_mul_first=False,
                    fuse_projection_weights= False,
                    blocks_per_ckpt =1,
                    tune_chunk_size=False
            )
            # self.input_proj = Linear(self.outdim, hidden_dim)
            self.output_proj = Linear(hidden_dim+zdim, self.outdim )
        else:
            self.decoder_ = ResidLinearMLP(zdim, layers, hidden_dim, self.outdim, activation)

        self.n_pix_cutoff=int(np.ceil(quality_ratio * self.sigma / self.pixel_size) * 2 + 1)    

        print("--\n AF DECODER PARAMETERS :")
        print("\t mode : %s"%("real space" if isinstance(self, AFDecoderReal) else "reciprocal"))
        print("\t domain : %s"%(domain))
        print("\t pair_stack : %s"%str(pair_stack))
        print("\t integration diameter set to %i"%((self.n_pix_cutoff)))
        print("--\n")


    def forward(self, crd_lattice, z):

        struct = self.structure_decoder(z)

        # Apply initial transform
        crd = struct_to_crd(struct, ca=not self.all_atom)
        crd = crd @ self.rot_init + self.trans_init[..., None, :]


        if self.domain == "hartley":

            y_recon = img_ht_lattice(
                crd=crd, 
                crd_lattice=crd_lattice, 
                sigma = self.sigma, 
                pixel_size=self.pixel_size
            )
        else:
            y_recon = img_ft_lattice(
                crd=crd, 
                crd_lattice=crd_lattice, 
                sigma = self.sigma, 
                pixel_size=self.pixel_size
            )
        return y_recon, struct

    def structure_decoder(self, z):
        inplace_safe = not (self.training or torch.is_grad_enabled())

        batch_dim = z.shape[:-1]
        embedding_expand = {
            k: v.unsqueeze(0).expand(batch_dim+ tuple(-1 for _ in v.shape)) for k, v in self.embeddings.items()
        }

        # [*, N**2, Zdim]
        pair_update = z.unsqueeze(-2).repeat(1, self.res_size **2, 1) 

        if self.pair_stack : 
            pos_mask = embedding_expand["seq_mask"]
            pair_mask = pos_mask[..., None] * pos_mask[..., None, :]

        
            pair_update = pair_update.reshape(batch_dim + (self.res_size, self.res_size, self.zdim))
            pair_update = torch.cat ((embedding_expand["pair"], pair_update) , dim=-1)

            # pair_update =self.input_proj(pair_update)

            pair_update = self.decoder_(
                pair_update[..., None, :,:,:],
                mask=pair_mask[..., None, :,:],
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                # use_flash=self.globals.use_flash,
                inplace_safe=not (self.training or torch.is_grad_enabled()),
                _mask_trans=self.config._mask_trans,
            )
            pair_update =self.output_proj(pair_update[..., -1, :, :, :])
            pair_update = add(embedding_expand["pair"], pair_update, inplace_safe)


        else:
            pair_update = self.decoder_(pair_update)

            # [*, N, N, Pdim]
            pair_update = pair_update.reshape(batch_dim + (self.res_size, self.res_size, self.outdim))
            
            # [*, N, N, Pdim]
            pair_update = add(embedding_expand["pair"], pair_update, inplace_safe)

        structure_input = {
            "pair": pair_update,
            "single": embedding_expand["single"]
        }
                # Predict 3D structure
        outputs = {}
        outputs["sm"] = self.structure_module(
            structure_input,
            embedding_expand["aatype"],
            mask=embedding_expand["seq_mask"],
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], embedding_expand
        )
        outputs["final_atom_mask"] = embedding_expand["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]


        for k, v in embedding_expand.items():
            if k not in outputs : 
                outputs[k] = v

        return outputs
    
    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        """
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated

        z = torch.tensor(zval[None], device=coords.device, dtype=coords.dtype)
        struct = self.structure_decoder(z)

        # Apply initial transform
        crd = struct_to_crd(struct, not self.all_atom)
        crd = crd @ self.rot_init + self.trans_init[..., None, :]

        # vol = vol_real(crd, grid_size = D, sigma = self.sigma, pixel_size=self.pixel_size)


        vox_loc, vox_mask = get_voxel_mask(crd, D, self.pixel_size,  self.n_pix_cutoff)
        vol = vol_real_mask(
            crd, 
            vox_loc, 
            vox_mask, 
            D, 
            self.sigma, 
            self.pixel_size
        )

        return vol[-1].detach(), struct
    

def struct_to_crd(struct, ca=True):
    """
    Conversion of coordinates in AF 3-7 format + mask to array of Nx3
    A bit complicated just to apply mask on AF coordinates crd[mask]
    """
    crd = struct["final_atom_positions"]
    mask = struct["final_atom_mask"]
    batch_dim =crd.shape[:-3] 
    if ca:
        crd = crd[..., 1, :]
        mask = mask[..., 1]
        crd = (crd[mask == 1.0])
    else:
        flat_idx_shape = batch_dim + (crd.shape[-3]*crd.shape[-2],)
        crd = (crd * mask[..., None]).reshape(flat_idx_shape+(3,))
        crd = crd[mask.reshape(flat_idx_shape) == 1.0]
    crd = crd.reshape(
        batch_dim + (reduce(operator.mul, crd.shape, 1)//(reduce(operator.mul, batch_dim, 1)*3), 3)
    )
    return crd




#
class AFDecoderReal(AFDecoder):
    def __init__(self, 
                 **kwargs,
        ):
        super(AFDecoderReal, self).__init__(**kwargs)

    def forward(self, rot, z):

        struct = self.structure_decoder(z)

        # Apply initial transform
        crd = struct_to_crd(struct, ca=not self.all_atom)
        crd = crd @ self.rot_init + self.trans_init[..., None, :]

        # print(crd.shape)
        # print(rot.shape)
        crd = crd @ rot.transpose(-1,-2)

        # y_recon_real = img_real(
        #     crd=crd, 
        #     grid_size=self.lattice_size, 
        #     sigma = self.sigma, 
        #     pixel_size=self.pixel_size
        # )
        pix_loc, pix_mask = get_pixel_mask(
            crd, 
            self.lattice_size, 
            self.pixel_size, 
            self.n_pix_cutoff
        )
        y_recon_real = img_real_mask(
            crd, 
            pix_loc, 
            pix_mask, 
            self.lattice_size, 
            self.sigma, 
            self.pixel_size
        )
        y_recon = fft.fftshift(fft.fftn(fft.fftshift(y_recon_real), dim=(-1,-2)))

        if self.domain == "hartley":
            y_recon = y_recon.real-y_recon.imag

        return y_recon, struct, y_recon_real


def resume_ckpt(resume_from_ckpt, model, config):
    sd = torch.load(resume_from_ckpt)
    if "model_state_dict" not in sd :
        sd = convert_deprecated_v1_keys(sd)
        incompatible_keys = model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd["model_state_dict"], strict=False)
    return model