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

from openfold.utils.import_weights import import_jax_weights_, import_openfold_weights_
from openfold.utils.script_utils import get_model_basename
import os
from openfold.utils.import_weights import  process_translation_dict, assign, Param, ParamType

from openfold.data import mmcif_parsing
from openfold.data.data_pipeline import add_assembly_features, make_sequence_features, convert_monomer_features
from openfold.model.model import AlphaFold
from flexfold.core import output_single_pdb
from openfold.data.data_pipeline import  DataPipelineMultimer, DataPipeline


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

        # added paramrs
        initial_pose_path = None,
        embedding_path="./output/cryofold/AK_embeddings.pt",
        af_checkpoint_path = "none",
        sigma = 2.5,
        pixel_size = 2.2,
        quality_ratio = 5,
        real_space = True,
        all_atom=True,
        pair_stack = False,
        target_file=None,
        is_multimer=False,
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == "conv":
            # self.encoder = ConvEncoder(qdim, zdim * 2)
            self.encoder = PyramidConvEncoder(qdim, zdim * 2, False, max_hidden_blocks=qlayers)
        elif encode_mode == "resid":
            self.encoder = ResidLinearMLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,  # nlayers  # hidden_dim  # out_dim
            )
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        self.encode_mode = encode_mode

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
            target_file=target_file,
            domain = domain,
            is_multimer=is_multimer,
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

            initial_pose_path = c["initial_pose_path"],
            embedding_path=c["embedding_path"],
            af_checkpoint_path = c["af_checkpoint_path"],
            sigma = c["sigma"],
            pixel_size = c["pixel_size"],
            quality_ratio = c["quality_ratio"],
            real_space = c["real_space"],
            all_atom = c["all_atom"],
            pair_stack = c["pair_stack"] if "pair_stack" in c else False,
            target_file= c["target_file"] if "target_file" in c else None,
            is_multimer = c["is_multimer"]
        )
        if weights is not None:
            ckpt = torch.load(weights)
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
# Optional self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)  # [seq_len, batch, channels]
        x_attn = self.attn(x_flat, x_flat, x_flat)[0]
        x_attn = self.norm(x_attn)
        x_attn = x_attn.permute(1, 2, 0).view(B, C, H, W)
        return x_attn    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)  # smoother than LeakyReLU

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        skip = self.skip(x)
        return self.act(out + skip)


# Modernized ConvEncoder
class PyramidConvEncoder(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=8, use_attention=False, min_spatial=4, max_hidden_blocks=4):
        """
        Pyramid ConvEncoder that adapts to arbitrary input image sizes.
        
        Args:
            hidden_dim: Base number of channels for first block.
            out_dim: Output feature dimension.
            use_attention: Whether to apply SelfAttention at the last layer.
            min_spatial: Minimum spatial size before global pooling.
        """
        super().__init__()
        self.use_attention = use_attention
        self.min_spatial = min_spatial

        # initial input channels
        in_channels = 1
        current_hidden = hidden_dim
        self.layers = nn.ModuleList()

        # Add pyramid of residual blocks until feature map ≤ min_spatial
        # We'll add 4 as a default base, but can adapt dynamically if desired
        print("=========================================================")
        print("Pyramidal encoder architecture : ")
        for i in range(max_hidden_blocks):
            self.layers.append(ResidualBlock(in_channels, current_hidden, stride=2))
            print("Conv block %i (input img size %i) : %i -> %i"%(i+1,(4* 2**(max_hidden_blocks-i)), in_channels, current_hidden))
            in_channels = current_hidden
            current_hidden *= 2
        print("=========================================================")



        self.attn = SelfAttention(in_channels) if use_attention else nn.Identity()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)



    def forward(self, x):
        B, N2 = x.shape
        N = int(N2**0.5)
        x = x.reshape((B, 1, N, N))

        # pass through pyramid blocks
        for layer in self.layers:
            # stop early if spatial dimensions are small enough
            if x.shape[-2] <= self.min_spatial and x.shape[-1] <= self.min_spatial:
                break
            x = layer(x)

        x = self.attn(x)
        x = self.proj(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

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
        target_file: str=None,
        is_multimer: bool =False,
        domain: str = "hartley"
):
    embeddings = torch.load(embedding_path,map_location="cpu")

    initial_pose = torch.load(initial_pose_path)

    print("\nSuccessfully loaded %s"%initial_pose_path)
    print("Rotation : ")
    print(initial_pose["R"])
    print("Translation : ")
    print(initial_pose["T"])
    print()


    config_preset = get_model_basename(af_checkpoint_path)
    if config_preset.startswith("params_") :
         config_preset[7:]
    elif len(config_preset) >= 2 and config_preset[-2] == "_" and config_preset[-1].isdigit():
        config_preset = config_preset[:-2]

    config = model_config(
        config_preset, 
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
        "target_file": target_file,
        "domain" : domain,
        "is_multimer":is_multimer,
    }

    if real_space:
        model = AFDecoderReal(**afdecoder_args)
    else:
        model = AFDecoder(**afdecoder_args)

    model = import_weights(model,af_checkpoint_path)

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
                target_file,
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
        self.target_file=target_file
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
            "asym_id": [NUM_RES],
        }
    

        embeddings = {k: v for k, v in embeddings.items() if k in embeddings_keys.keys()}

        if target_file is not None:
            target_feats = get_target_feats(
                target_file, 
                DataPipelineMultimer(DataPipeline(None)) if is_multimer else DataPipeline(None)
            )


        def make_gt(feats):
            feats["all_atom_positions"] = torch.tensor(target_feats ["all_atom_positions"]) if target_file is not None else feats["final_atom_positions"] 
            feats["all_atom_mask"] = feats["final_atom_mask"] 
            return feats
    
        fc = [
            # data_transforms.make_fixed_size(embeddings_keys,0,0,500,0),
            make_gt,
            data_transforms.make_atom14_positions,
            data_transforms.atom37_to_frames,
            data_transforms.atom37_to_torsion_angles(""),
            data_transforms.make_pseudo_beta(""),
            data_transforms.get_backbone_frames,
            data_transforms.get_chi_angles,
        ]
        for f in fc:
            embeddings = f(embeddings)

        self.embeddings = BufferDict(embeddings)

        self.res_size = self.embeddings["pair"].shape[-2]
        self.outdim = self.embeddings["pair"].shape[-1]

        self.zdim = zdim
        self.hidden_dim=hidden_dim

        if self.pair_stack : 

            self.input_proj = Linear(zdim, hidden_dim )
            self.embedding_proj = Linear(self.outdim, hidden_dim )
            self.decoder_ = TemplatePairStack(
                    c_t=hidden_dim,
                    c_hidden_tri_att= hidden_dim//4,
                    c_hidden_tri_mul= hidden_dim//2,
                    no_blocks= layers,
                    no_heads= 4,
                    pair_transition_n= layers,
                    dropout_rate= 0.15,
                    tri_mul_first=False,
                    fuse_projection_weights= False,
                    blocks_per_ckpt =1,
                    tune_chunk_size=False
            )
            self.output_proj = Linear(hidden_dim, self.outdim )
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

        if self.pair_stack : 
            pos_mask = embedding_expand["seq_mask"]
            pair_mask = pos_mask[..., None] * pos_mask[..., None, :]

            # [B, 1, zdim]
            pair_update = self.input_proj(z)

            # [B, N, N, hidden_dim]
            pair_update = pair_update.unsqueeze(-2).repeat(1, self.res_size **2, 1) 
            pair_update = pair_update.reshape(batch_dim + (self.res_size, self.res_size, self.hidden_dim))

            # [B, N, N, hidden_dim]
            embedding_update = self.embedding_proj(embedding_expand["pair"])
            pair_update = add(embedding_update, pair_update, inplace_safe)

            # [B, N, N, hidden_dim]
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

            # [B, N, N, outdim]
            pair_update =self.output_proj(pair_update[..., -1, :, :, :])
            pair_update = add(embedding_expand["pair"], pair_update, inplace_safe)


        else:
            # [*, N**2, Zdim]
            pair_update = z.unsqueeze(-2).repeat(1, self.res_size **2, 1) 

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

        crd = crd @ rot.transpose(-1,-2)

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
        y_recon = fft.fft2_center(y_recon_real)

        return y_recon, struct



def import_weights(model, ckpt_path):
    ext = os.path.splitext(ckpt_path)[1] 

    if ext == ".npz":
        model_basename = get_model_basename(ckpt_path)
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_sm(
            model, ckpt_path, version=model_version
        )
    elif ext == ".pt":
        d = torch.load(ckpt_path)

        if "ema" in d:
            # The public weights have had this done to them already
            d = d["ema"]["params"]
        import_openfold_weights_sm(model=model, state_dict=d)
    else:
        raise
    return model


def import_openfold_weights_sm(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        converted_state_dict = convert_deprecated_v1_keys(state_dict)
        model.load_state_dict(converted_state_dict, strict=False)


def import_jax_weights_sm(model, npz_path, version="model_1"):
    data = np.load(npz_path)
    translations = generate_translation_dict_sm(model, version, is_multimer=("multimer" in version))

    # Flatten keys and insert missing key prefixes
    flat = process_translation_dict(translations)

    # Sanity check
    keys = list(data.keys())
    flat_keys = list(flat.keys())
    incorrect = [k for k in flat_keys if k not in keys]
    missing = [k for k in keys if k not in flat_keys]
    # print(f"Incorrect: {incorrect}")
    # print(f"Missing: {missing}")

    assert len(incorrect) == 0
    # assert(sorted(list(flat.keys())) == sorted(list(data.keys())))

    # Set weights
    assign(flat, data)

def generate_translation_dict_sm(model, version, is_multimer=False):

    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))
    LinearBias = lambda l: (Param(l))
    LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))
    LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))

    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }

    LinearParamsMHA = lambda l: {
        "weights": LinearWeightMHA(l.weight),
        "bias": LinearBiasMHA(l.bias),
    }

    LayerNormParams = lambda l: {
        "scale": Param(l.weight),
        "offset": Param(l.bias),
    }

    IPAParams = lambda ipa: {
        "q_scalar": LinearParams(ipa.linear_q),
        "kv_scalar": LinearParams(ipa.linear_kv),
        "q_point_local": LinearParams(ipa.linear_q_points.linear),
        "kv_point_local": LinearParams(ipa.linear_kv_points.linear),
        "trainable_point_weights": Param(
            param=ipa.head_weights, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.linear_b),
        "output_projection": LinearParams(ipa.linear_out),
    }

    PointProjectionParams = lambda pp: {
        "point_projection": LinearParamsMHA(
            pp.linear,
        ),
    }

    IPAParamsMultimer = lambda ipa: {
        "q_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_q.weight,
            ),
        },
        "k_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_k.weight,
            ),
        },
        "v_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_v.weight,
            ),
        },
        "q_point_projection": PointProjectionParams(
            ipa.linear_q_points
        ),
        "k_point_projection": PointProjectionParams(
            ipa.linear_k_points
        ),
        "v_point_projection": PointProjectionParams(
            ipa.linear_v_points
        ),
        "trainable_point_weights": Param(
            param=ipa.head_weights, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.linear_b),
        "output_projection": LinearParams(ipa.linear_out),
    }

    def FoldIterationParams(sm):
        d = {
            "invariant_point_attention": 
                IPAParamsMultimer(sm.ipa) if is_multimer else IPAParams(sm.ipa),
            "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
            "transition": LinearParams(sm.transition.layers[0].linear_1),
            "transition_1": LinearParams(sm.transition.layers[0].linear_2),
            "transition_2": LinearParams(sm.transition.layers[0].linear_3),
            "transition_layer_norm": LayerNormParams(sm.transition.layer_norm),
            "affine_update": LinearParams(sm.bb_update.linear),
            "rigid_sidechain": {
                "input_projection": LinearParams(sm.angle_resnet.linear_in),
                "input_projection_1": 
                    LinearParams(sm.angle_resnet.linear_initial),
                "resblock1": LinearParams(sm.angle_resnet.layers[0].linear_1),
                "resblock2": LinearParams(sm.angle_resnet.layers[0].linear_2),
                "resblock1_1": 
                    LinearParams(sm.angle_resnet.layers[1].linear_1),
                "resblock2_1": 
                    LinearParams(sm.angle_resnet.layers[1].linear_2),
                "unnormalized_angles": 
                    LinearParams(sm.angle_resnet.linear_out),
            },
        }

        if(is_multimer):
            d.pop("affine_update")
            d["quat_rigid"] = {
                "rigid": LinearParams(
                   sm.bb_update.linear
                )
            }

        return d

    if(not is_multimer):
        translations = {
            "structure_module": {
                "single_layer_norm": LayerNormParams(
                    model.structure_module.layer_norm_s
                ),
                "initial_projection": LinearParams(
                    model.structure_module.linear_in
                ),
                "pair_layer_norm": LayerNormParams(
                    model.structure_module.layer_norm_z
                ),
                "fold_iteration": FoldIterationParams(model.structure_module),
            },
        }
    else:
        translations = {
            "structure_module": {
                "single_layer_norm": LayerNormParams(
                    model.structure_module.layer_norm_s
                ),
                "initial_projection": LinearParams(
                    model.structure_module.linear_in
                ),
                "pair_layer_norm": LayerNormParams(
                    model.structure_module.layer_norm_z
                ),
                "fold_iteration": FoldIterationParams(model.structure_module),
            },
        }
    return translations


def get_target_feats(mmcif_file,data_processor):

    with open(mmcif_file, 'r') as f:
        mmcif_string = f.read()

    mmcif_object = mmcif_parsing.parse(
        file_id="1HZH", mmcif_string=mmcif_string
    )

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with at the alignment stage.
    if mmcif_object.mmcif_object is None:
        raise list(mmcif_object.errors.values())[0]

    mmcif_object = mmcif_object.mmcif_object

    all_chain_features = {}
    for chain_id, seq in mmcif_object.chain_to_seqres.items():
        desc= "_".join([mmcif_object.file_id, chain_id])
        input_sequence = mmcif_object.chain_to_seqres[chain_id]
        num_res = len(input_sequence)

        mmcif_feats = {}

        mmcif_feats.update(
            make_sequence_features(
                sequence=input_sequence,
                description=desc,
                num_res=num_res,
            )
        )
        mmcif_feats.update(data_processor.get_mmcif_features(mmcif_object, chain_id))

        mmcif_feats = convert_monomer_features(
            mmcif_feats,
            chain_id=desc
        )

        all_chain_features[desc] = mmcif_feats

    all_chain_features = add_assembly_features(all_chain_features)

    merge_feats = {}
    for f in ["asym_id","all_atom_positions","all_atom_mask","residue_index","aatype"]:
        merge_feats[f] = np.concatenate([v[f] for k,v in all_chain_features.items()], axis=0)
    return merge_feats
