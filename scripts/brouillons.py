import torch
from flexfold.core import struct_to_pdb
from openfold.utils.tensor_utils import tensor_tree_map
import numpy as np 
from openfold.np import residue_constants, protein
from openfold.config import model_config
from openfold.model.model import AlphaFold









struct = torch.load("/home/vuillemr/cryofold/cryobench_IgD/pred2/embeddings.pt")
struct = tensor_tree_map(lambda x: x.detach().cpu().numpy(), struct)
struct_to_pdb(struct,"/home/vuillemr/cryofold/cryobench_IgD/test.pdb" )


config = model_config(
    "model_3_multimer_v3", 
    train=True, 
    low_prec=False,
) 
model = AlphaFold(config)



structure_module = StructureModule(
            is_multimer=is_multimer,
            **self.config["structure_module"],
        )
structure_input = {
    "pair": struct["pair"],
    "single": struct["single"]
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










from cryodrgn import utils, config
z = utils.load_pkl("data/cryofold/particlesSNR1.0/run_fourier_inf_baseline/z.6.pkl")
zdim = z.shape[1]


from cryodrgn import utils, config
from flexfold.models import HetOnlyVAE
from flexfold import dataset




dataset.ImageDataset(
            mrcfile="data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/099_particles_128.mrcs",
            lazy="False",
            norm=None,
            invert_data=False,
            ind=None,
            keepreal=False,
            window=True,
            datadir=None,
            window_r=0.85,
            max_threads=0,
        )


cfg = config.load("data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/config.yaml")
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg, "data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run/weights.0.pkl", device="cuda:0")
model.eval()







import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
from Bio.PDB.MMCIFParser import MMCIFParser
embeddings=  torch.load("data/cryofold/spike-md/embeddings.pt", map_location="cuda:0")

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("pdb", "data/cryofold/spike-md/6VSB.cif")
model = next(structure.get_models())  # usually only 1 model
# Extract all residues (modeled)
modeled = set()
for chain in model:
    if chain.id == "A":
        for res in chain:
            if res.id[0] == " ":  # ignore hetero/water
                modeled.add(res.id[1])  # resseq number

modeled = torch.tensor(list(modeled), device="cuda:0")


resisd = embeddings["residue_index"]

crop = (resisd<=(1146-1)) * (resisd>=(27-1))
mask = torch.tensor([1 if i in modeled else 0 for i in embeddings["residue_index"]], device="cuda:0")

output_single_pdb(
    embeddings["final_atom_positions"][crop].cpu().numpy(),
    embeddings["aatype"][crop].cpu().numpy(),
    (mask[..., None]*  embeddings["final_atom_mask"])[crop].cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    embeddings["asym_id"][crop].cpu().numpy(),
    embeddings["residue_index"][crop].cpu().numpy(),

)

output_single_pdb(
    embeddings["final_atom_positions"][crop].cpu().numpy(),
    embeddings["aatype"][crop].cpu().numpy(),
    (embeddings["final_atom_mask"])[crop].cpu().numpy(),
    "data/cryofold/spike-md/test_unmask.pdb",
    embeddings["asym_id"][crop].cpu().numpy(),
    embeddings["residue_index"][crop].cpu().numpy(),

)

embeddings_new = {
    "aatype": embeddings["aatype"][crop],
    "seq_mask": mask[crop],
    "pair": embeddings["pair"][crop, :, :][:, crop, :],
    "single":  embeddings["single"][crop, :],
    "final_atom_mask": (mask[..., None]* embeddings["final_atom_mask"])[crop, :],
    "final_atom_positions":embeddings["final_atom_positions"][crop, :,:],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"][crop, :],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"][crop, :],
    "atom37_atom_exists":embeddings["atom37_atom_exists"][crop, :],
    "atom14_atom_exists":embeddings["atom14_atom_exists"][crop, :],
    "residue_index": embeddings["residue_index"][crop],
    "asym_id":  embeddings["asym_id"][crop],
}

output_single_pdb(
    embeddings_new["final_atom_positions"].cpu().numpy(),
    embeddings_new["aatype"].cpu().numpy(),
    (embeddings_new["final_atom_mask"]).cpu().numpy(),
    "data/cryofold/spike-md/test.pdb",
    embeddings_new["asym_id"].cpu().numpy(),
    embeddings_new["residue_index"].cpu().numpy(),

)

torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/spike-md/embeddings_crop_mask.pt")


import torch
from openfold.np.residue_constants import restypes
from flexfold.core import output_single_pdb
embeddings=  torch.load("data/cryofold/jillsData/pred2/embeddings.pt")
embeddings_new = {
    "aatype": embeddings["aatype"],
    "seq_mask": embeddings["seq_mask"],
    "pair": embeddings["pair"],
    "single":  embeddings["single"],
    "final_atom_mask": embeddings["final_atom_mask"],
    "final_atom_positions":embeddings["final_atom_positions"],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"],
    "atom37_atom_exists":embeddings["atom37_atom_exists"],
    "atom14_atom_exists":embeddings["atom14_atom_exists"],
    "residue_index": embeddings["residue_index"],
    "asym_id":  embeddings["asym_id"],
}
torch.save({k: v.cpu() for k, v in embeddings_new.items()}, "data/cryofold/jillsData/pred2/embeddings_fixed.pt")


import pandas as pd
import io

def read_star_multi(star_path):
    blocks = {}
    with open(star_path, "r") as f:
        lines = f.readlines()

    current_block = None
    headers = []
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("data_"):
            # flush previous block if any
            if current_block and headers:
                df = pd.read_csv(
                    io.StringIO("\n".join(data_lines)),
                    delim_whitespace=True,
                    names=headers,
                )
                blocks[current_block] = df.to_dict(orient="list")
            # reset for new block
            current_block = line
            headers, data_lines = [], []
        elif line.startswith("_rln"):
            headers.append(line.split()[0])
        elif headers and line and not line.startswith("#"):
            data_lines.append(line)

    # flush last block
    if current_block and headers:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            delim_whitespace=True,
            names=headers,
        )
        blocks[current_block] = df.to_dict(orient="list")


    print("========================== Summary ==========================")
    for k,v in blocks.items():
        print(k)
        print("|")
        for k2,v2 in v.items():
            print("|--> %s, len=%i, dtype=%s"%(k2, len(v2), type(next(iter(v2)))))
        print("")

    return blocks

def write_star_from_dict(star_dict, out_path, data_block="data_particles"):
    """
    Write a RELION .star file from a dict of lists.

    Parameters
    ----------
    star_dict : dict
        Keys are RELION column names (e.g. '_rlnImageName'),
        values are lists of column entries (all same length).
    out_path : str
        Where to write the .star file.
    data_block : str
        Name of the data block (default: "data_particles").
    """
    keys = list(star_dict.keys())
    n_rows = len(next(iter(star_dict.values())))  # length of first column

    with open(out_path, "w") as f:
        # block header
        f.write(f"{data_block}\n\n")
        f.write("loop_\n")
        # headers with column numbers
        for i, k in enumerate(keys, start=1):
            f.write(f"{k} #{i}\n")

        # write rows
        for row_idx in range(n_rows):
            row = [str(star_dict[k][row_idx]) for k in keys]
            f.write(" ".join(row) + "\n")


star = read_star_multi("/home/vuillemr/flexfold/data/cryofold/jillsData/particles/particles_ctf.star")


keywords = [
    ["_anglePsi","_rlnAnglePsi"],
    ["_angleRot","_rlnAngleRot"],
    ["_angleTilt","_rlnAngleTilt"],
    ["_image","_rlnImageName"],
    ["_shiftX","_rlnOriginXAngst"],
    ["_shiftY","_rlnOriginYAngst"],
    ["_ctfVoltage","_rlnVoltage"],
    ["_ctfSphericalAberration","_rlnSphericalAberration"],
    ["_ctfSamplingRate","_rlnDetectorPixelSize"],
    ["_magnification","_rlnMagnification"],
    ["_ctfDefocusU","_rlnDefocusU"],
    ["_ctfDefocusV","_rlnDefocusV"],
    ["_ctfQ0","_rlnAmplitudeContrast"]
]

xmp2rln_kw = {k:v for k,v in keywords}
rln2xmp_kw = {v:k for k,v in keywords}

def rln2xmp(star_dict):
    return {rln2xmp_kw[k]:v for k,v in star_dict.items() if k in rln2xmp_kw}
    
def xmp2rln(star_dict):
    return {xmp2rln_kw[k]:v for k,v in star_dict.items() if k in rln2xmp_kw}
    


write_star_from_dict(rln2xmp(star["data_particles"]), "/home/vuillemr/flexfold/data/cryofold/jillsData/particles/particles.xmp")









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

def parse_lightning_csv(path):
    df = pd.read_csv(path)

    # Always keep step + epoch columns if present
    base_cols = [c for c in ["step", "epoch"] if c in df.columns]

    # Classify metrics
    step_metrics  = [c for c in df.columns if c.endswith("_step")]
    epoch_metrics = [c for c in df.columns if c.endswith("_epoch")]

    # Separate val vs train
    val_step_cols   = [c for c in step_metrics if c.startswith("val")]
    train_step_cols = [c for c in step_metrics if c not in val_step_cols]

    val_epoch_cols   = [c for c in epoch_metrics if c.startswith("val")]
    train_epoch_cols = [c for c in epoch_metrics if c not in val_epoch_cols]

    # Build DataFrames (dropping rows where all relevant metrics are NaN)
    train_step_df  = df[base_cols + train_step_cols].dropna(how="all", subset=train_step_cols).reset_index(drop=True)
    train_epoch_df = df[base_cols + train_epoch_cols].dropna(how="all", subset=train_epoch_cols).reset_index(drop=True)
    val_step_df    = df[base_cols + val_step_cols].dropna(how="all", subset=val_step_cols).reset_index(drop=True)
    val_epoch_df   = df[base_cols + val_epoch_cols].dropna(how="all", subset=val_epoch_cols).reset_index(drop=True)

    def get_epoch_smooth():
        epoch_float = []
        for e, group in df.groupby("epoch"):
            steps = group["step"].values
            steps_per_epoch = steps.max() - steps.min() + 1
            epoch_frac = e + (steps - steps.min()) / steps_per_epoch
            epoch_float.extend(epoch_frac)

    return train_step_df, train_epoch_df, val_step_df, val_epoch_df

   

# Example usage


infile = "/home/vuillemr/cryofold/particlesSNR1.0/run_test/metrics.csv"
outfile = "/home/vuillemr/cryofold/particlesSNR1.0/run_test/metrics.svg"

train_step, train_epoch, val_step, val_epoch = parse_lightning_csv(infile)

movavg = lambda arr,w: np.convolve(
    np.nan_to_num(arr), np.ones(w), 'valid'
) / np.convolve(~np.isnan(arr), np.ones(w), 'valid')
movavg_step = lambda arr,w: arr[:-(w-1)]*(len(arr)/(len(arr)-w))

losses=["data_loss", "chi_loss", "viol_loss","pose_rot", "kld", "loss"]
col = "tab:blue"
valcol = "tab:green"
nrows = 2
ncols=3
fig, ax = plt.subplots(nrows,ncols, figsize=(10,5), layout="constrained")
for x in range(nrows):
    for y in range(ncols):
        ii = x *ncols + y
        if ii>=len(losses):
            break

        if not losses[ii]+"_step" in train_step:
            break

        loss = train_step[losses[ii]+"_step"]
        step = train_step["step"] * (train_step["epoch"].max() - train_step["epoch"].min()) / (train_step["step"].max() - train_step["step"].min())
        if len(step)>50:
            w = min(len(step)//10,10)
            ax[x,y].plot(step, loss, alpha=0.5, c=col)
            ax[x,y].plot(movavg_step(step,w), movavg(loss,w), label = "training", c=col)
        else:
            ax[x,y].plot(step, loss, label = "training", c=col)

        ax[x,y].set_xlabel("epoch")
        ax[x,y].set_ylabel(losses[ii])

        if ("val_" + losses[ii]+"_epoch") in val_epoch:
            loss = val_epoch["val_" +losses[ii]+"_epoch"]
            step = val_epoch["epoch"]
            if len(step)>50:
                w = min(len(step)//10,10)
                ax[x,y].plot(step, loss, alpha=0.5, c=valcol)
                ax[x,y].plot(movavg_step(step,w), movavg(loss,w), label = "validation", c=valcol)
            else:
                ax[x,y].plot(step, loss, label = "validation", c=valcol)

ax[0,0].legend()
fig.savefig(outfile, dpi=300)

plt.close(fig)

from openfold.model.primitives import Attention
from openfold.model.triangular_multiplicative_update import (FusedTriangleMultiplicationOutgoing,FusedTriangleMultiplicationIncoming,
                                                             TriangleMultiplicationOutgoing,TriangleMultiplicationIncoming)
from openfold.model.pair_transition import PairTransition
import torch
from openfold.model.primitives import Linear, LayerNorm, Attention
from openfold.utils.chunk_utils import chunk_layer
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
import torch.nn as nn
from typing import Optional, List
from functools import partialmethod, partial
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.utils.tensor_utils import add
from openfold.utils.checkpointing import checkpoint_blocks

class TriangleCrossAttention(nn.Module):
    def __init__(
        self, c_q, c_kv, c_hidden, no_heads, inf=1e9
    ):
        super(TriangleCrossAttention, self).__init__()
        self.c_q = c_q
        self.c_kv = c_kv
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.layer_norm = LayerNorm(self.c_q)
        # self.linear = Linear(self.c_q, self.no_heads, bias=False, init="normal")
        self.mha = Attention(
            self.c_q, self.c_kv, self.c_kv, self.c_hidden, self.no_heads
        )

    def forward(self, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        **kwargs
    ) -> torch.Tensor:

        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        print("Input x ->",x.requires_grad)
        print("Input z ->",z.requires_grad)
        # [*, I, J, C_in]
        x = self.layer_norm(x)        
        print("Layer Norm x->",x.requires_grad)
        print("Layer Norm z->",z.requires_grad)


        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, :, None]
        biases = [mask_bias]
        print("Mask ->",mask.requires_grad)
 
        x = self.mha(
                q_x=x, 
                kv_x=z, 
                biases=biases, 
                **kwargs
        )

        print("Output    ->",x.requires_grad)
        print()
        print()
        return x



class CryoFormerBlock(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_latent:int,
        c_hidden: int,
        no_heads: int,
        transition_n: int,
        dropout_rate: float,
        inf: float,
        eps: float
    ):
        super(CryoFormerBlock, self).__init__()

        self.tri_att_start = TriangleCrossAttention(
            c_z,
            c_latent,
            c_hidden,
            no_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleCrossAttention(
            c_z,
            c_latent,
            c_hidden,
            no_heads,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )
        self.ps_dropout_row_layer = DropoutRowwise(dropout_rate)

    def forward(self,
        z: torch.Tensor,
        latent:torch.Tensor,
        pair_mask: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        
        print("BLOCK ")

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        latent,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        latent,
                        mask=pair_mask.transpose(-1, -2),
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z, mask=None, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
        )

        return z

class CryoFormerStack(nn.Module):
    def __init__(
        self,
        c_z,
        c_latent,
        c_hidden,
        no_heads,
        no_blocks,
        transition_n,
        dropout_rate,
        blocks_per_ckpt,
        inf=1e9,
        eps = 1e-8,
    ):

        super(CryoFormerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = CryoFormerBlock(
                c_z=c_z,
                c_latent=c_latent,
                c_hidden=c_hidden,
                no_heads=no_heads,
                transition_n=transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_z)

    def forward(
        self,
        z: torch.tensor,
        latent: torch.tensor,
        mask: torch.tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ):

        blocks = [
            partial(
                b,
                latent=latent,
                pair_mask=mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
            for b in self.blocks
        ]

        z, = checkpoint_blocks(
            blocks=blocks,
            args=(z,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        z = self.layer_norm(z)

        return z











# make sure we're on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B = 1
N = 214
c_q = 64
c_kv = 8
c_hidden = 16
no_heads = 4
transition_n = 2
pair_dropout = 0.15
no_blocks=2
blocks_per_ckpt= 1 

q_x = torch.zeros (1,1,N,N,c_q, device=device, requires_grad=False)
mask = torch.ones (B,1,N,N,     device=device, requires_grad=False)
kv_x = torch.zeros(B,1,1,1,c_kv,device=device, requires_grad=True)
mask_bias = (1e8* (mask - 1))[..., :, None, :, None]
biases = [mask_bias]
a = CryoFormerStack(c_q,c_kv, c_hidden, no_heads,no_blocks, transition_n, pair_dropout,blocks_per_ckpt).to(device)
# a = CryoFormerBlock(c_q,c_kv, c_hidden, no_heads, transition_n, pair_dropout, 1e8, 1e-9).to(device)
# a = Attention(c_q,c_kv,c_kv, c_hidden, no_heads).to(device)


# reset peak memory counter
torch.cuda.reset_peak_memory_stats(device)

# forward
out = a(q_x, kv_x, mask)
# out = a(q_x, kv_x, biases)


# backward
loss = out.pow(2).sum()
loss.backward()

print(c_kv.grad is None)






###########################################################################################################################""
import torch
from flexfold.models import get_target_feats
from openfold.data.data_pipeline import  DataPipelineMultimer, DataPipeline
import numpy as np

e = torch.load("../cryofold/cryobench_IgD/pred2/embeddings.pt", map_location="cuda:1")
e["asym_id"]

t = get_target_feats("../cryofold/cryobench_IgD/1HZH.cif",DataPipelineMultimer(DataPipeline(None)))
(t["asym_id"] ==4).sum()



def transpose_chains(t, transpose):
    asym_id = t["asym_id"]
    order = np.unique(t["asym_id"])[[i for i in transpose]]
    order_ind = np.concatenate([np.where(asym_id==i)[0] for i in order])
    new_asym_id = np.concatenate([(i+1)*np.ones((asym_id==o).sum()) for i,o in enumerate(order)])
    t["all_atom_positions"] = t["all_atom_positions"][order_ind]
    t["aatype"] = t["aatype"][order_ind]
    t["asym_id"] = new_asym_id
    return t

t = transpose_chains(t, (1,0,3,2)) 

embeddings_new = {
    "aatype": embeddings["aatype"][crop],
    "seq_mask": mask[crop],
    "pair": embeddings["pair"][crop, :, :][:, crop, :],
    "single":  embeddings["single"][crop, :],
    "final_atom_mask": (mask[..., None]* embeddings["final_atom_mask"])[crop, :],
    "final_atom_positions":embeddings["final_atom_positions"][crop, :,:],
    "residx_atom37_to_atom14":embeddings["residx_atom37_to_atom14"][crop, :],
    "residx_atom14_to_atom37":embeddings["residx_atom14_to_atom37"][crop, :],
    "atom37_atom_exists":embeddings["atom37_atom_exists"][crop, :],
    "atom14_atom_exists":embeddings["atom14_atom_exists"][crop, :],
    "residue_index": embeddings["residue_index"][crop],
    "asym_id":  embeddings["asym_id"][crop],
}

















import os
from openfold.data import mmcif_parsing
from openfold.data.data_pipeline import add_assembly_features, make_sequence_features, convert_monomer_features
from openfold.data.data_pipeline import  DataPipelineMultimer, DataPipeline
from dataclasses import replace

data_processor = DataPipelineMultimer(DataPipeline(None))
mmcif_file="/home/vuillemr/flexfold/data/cryofold/cryobench_IgD/1HZH.cif"
with open(mmcif_file, 'r') as f:
    mmcif_string = f.read()

mmcif_object = mmcif_parsing.parse(
    file_id="1HZH", mmcif_string=mmcif_string
)
mmcif_object = mmcif_object.mmcif_object

dir(mmcif_object)

print(mmcif_object.chain_to_seqres)
print(mmcif_object.file_id)
print(mmcif_object.seqres_to_structure)
print(mmcif_object.structure)
print(mmcif_object.raw_string)

mmcif_object =replace(mmcif_object,raw_string = "")
mmcif_object =replace(mmcif_object,header = {"resolution":1.0, "release_date":"01/10/2025"})


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



##### PDB #####
import collections
import dataclasses
import functools
import io
import json
import logging
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

from Bio import PDB
from Bio.Data import PDBData
import numpy as np
from Bio.PDB import PDBParser
from collections import defaultdict
from openfold.data.errors import MultipleChainsError
import openfold.np.residue_constants as residue_constants
from openfold.data.mmcif_parsing import _get_first_model, _get_header, _get_protein_chains, ResidueAtPosition, ResiduePosition, MmcifObject, ParsingResult
from Bio.PDB import PDBParser, PPBuilder

def parse_pdb(pdb_string):
    parser = PDB.PDBParser(QUIET=True)
    handle = io.StringIO(pdb_string)
    full_structure = parser.get_structure("", handle)
    first_model_structure = _get_first_model(full_structure)

    def pdb_chain_sequences(structure):  
        ppb = PPBuilder()
        chain_seqs = {}
        for chain in structure:
            seq = ""
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())  # concatenate fragments
            if len(seq)>= 1:
                chain_seqs[chain.id] = seq
        
        return chain_seqs

    def build_residue_dict(structure):
        chain_residues = defaultdict(dict)
        for chain in structure:
            chain_id = chain.id

            # Extract residues (ignore hetero/water unless desired)
            residues = [res for res in chain.get_residues() if res.id[0] == " "]
            if not residues:
                continue

            # Just linear counter 0..N-1
            for idx, res in enumerate(residues):
                resnum = res.id[1]
                icode = res.id[2].strip() if res.id[2] != " " else " "
                hetflag = res.id[0]

                pos = ResiduePosition(chain_id=chain_id,
                                        residue_number=resnum,
                                        insertion_code=icode)
                chain_residues[chain_id][idx] = ResidueAtPosition(
                    position=pos,
                    name=res.get_resname(),
                    is_missing=False,
                    hetflag=hetflag,
                )
        return dict(chain_residues)
    
    def assert_residue_order(res_dict):
        for chain_id, residues in res_dict.items():
            prev_num, prev_icode = None, " "
            for idx in sorted(map(int, residues.keys())):
                pos = residues[idx].position
                num, icode = pos.residue_number, pos.insertion_code

                if prev_num is not None:
                    # Ensure strictly increasing residue numbers (or same num with insertion code ordering)
                    assert (num > prev_num) or (num == prev_num and icode > prev_icode), \
                        f"Residues out of order in chain {chain_id} at index {idx}: {prev_num}{prev_icode} → {num}{icode}"

                prev_num, prev_icode = num, icode

    chain_seqs = pdb_chain_sequences(first_model_structure)
    res_dict = build_residue_dict(first_model_structure)
    assert_residue_order(res_dict)

    mmcif_object = MmcifObject(
        file_id="",
        header={"resolution":1.0, "release_date":"01/10/2025"},
        structure=first_model_structure,
        chain_to_seqres=chain_seqs,
        seqres_to_structure=res_dict,
        raw_string="",
    )
    return ParsingResult(mmcif_object=mmcif_object, errors=None)

pdb_file="/home/vuillemr/flexfold/data/cryofold/jillsData/particles/../MEK1DDGRA_ERK2T185V_ADP_AF3_refined_019.pdb"
# pdb_file="/home/vuillemr/flexfold/data/cryofold/cryobench_IgD/1HZH.pdb"
with open(pdb_file, 'r') as f:
    pdb_string = f.read()

mmcif_object = parse_pdb(pdb_string)
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



struct = torch.load("/home/vuillemr/flexfold/data/cryofold/jillsData/pred2/embeddings_fixed.pt")

merge_feats["asym_id"].shape

from Bio import pairwise2

def map_sequences(seq1, seq2):
    # Align seq1 to seq2 (global alignment, penalize gaps to keep structure aligned)
    alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -5, -0.5)
    aln1, aln2, score, start, end = alignments[0]

    mapping = []
    idx1, idx2 = 0, 0

    for a1, a2 in zip(aln1, aln2):
        if a1 != "-" and a2 != "-":   # match/mismatch → map index
            mapping.append( idx2)
            idx1 += 1
            idx2 += 1
        elif a1 != "-" and a2 == "-": # gap in seq2 → no mapping
            mapping.append(( -1))
            idx1 += 1
        elif a1 == "-" and a2 != "-": # gap in seq1 → seq2 advances
            idx2 += 1

    return mapping

from openfold.np import residue_constants as rc
from openfold.utils.tensor_utils import tensor_tree_map

s1 = merge_feats
s1 = {k:torch.tensor(v, dtype=torch.long) if np.issubdtype(v.dtype, np.integer) else torch.tensor(v, dtype=torch.float) for k,v in s1.items()}

s2 = struct



seq1 = "".join([rc.restypes_with_x[i] for i in s1["aatype"]])
seq2 = "".join([rc.restypes_with_x[i] for i in s2["aatype"]])

mapping = map_sequences(seq1, seq2)

    target_keys = ["asym_id","final_atom_positions","all_atom_mask","residue_index","aatype"]
    s1_mapped = {k:v.clone() for k,v in {k2:v2 for k2,v2 in s2.items() if k2 in target_keys}.items()}
    s1_mapped["all_atom_positions"] = s1_mapped["final_atom_positions"]
    del s1_mapped["final_atom_positions"]
    for k,v in s1_mapped.items():
        v[mapping] = s1[k] 

assert all(s1_mapped["aatype"] == s2["aatype"])
assert all(s1_mapped["asym_id"] == s2["asym_id"])
