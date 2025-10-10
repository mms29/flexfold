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





########################################################################""
import torch 
from flexfold.models import HetOnlyVAE
from cryodrgn import config
from cryodrgn.utils import load_pkl
import matplotlib.pyplot as plt
from cryodrgn.commands_utils.fsc import get_fsc_curve, get_fsc_thresholds,calculate_cryosparc_fscs
from cryodrgn.commands_utils.plot_fsc import create_fsc_plot
from cryodrgn.mrcfile import parse_mrc, write_mrc
import numpy as np
from Bio.PDB import PDBParser, Superimposer, is_aa
from io import StringIO
from openfold.utils.tensor_utils import tensor_tree_map
from flexfold.core import struct_to_pdb
import pickle 

config_file = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/config.yaml"
weight_file = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/weights.54.pkl"
z_file =  "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/z.54.pkl"
outfile = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr0.01/run/out.54.pkl"
##################################################""
# LOAD MODEL and WEIGHTS
##################################################""


def aligned_rmsd(structure1, structure2):
    def get_ca_atoms(structure):
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue, standard=False) and "CA" in residue:
                        atoms.append(residue["CA"])
        return atoms

    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)

    # Superimpose and calculate RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)  # This aligns atoms2 onto atoms1
    rmsd = sup.rms
    return rmsd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config.load(config_file)
D = cfg["lattice_args"]["D"]  # image size + 1
zdim = cfg["model_args"]["zdim"]
norm = [float(x) for x in cfg["dataset_args"]["norm"]]
model, lattice = HetOnlyVAE.load(cfg,weight_file, device=device)
model.eval()
z = load_pkl(z_file)
zdim = z.shape[-1]
parser = PDBParser(QUIET=True)

n_gt = 100
n_expand = 100

outputs = {
    "fsc": [],
    "fscauc": [],
    "fscres": [],
    "rmsd": []
}

for ind in range(n_gt):

    # GT VOLUME
    gt_vol_file = "../cryofold/AKMD/vols/out%s.mrc"%str(ind+1).zfill(6)
    vol,header =parse_mrc(gt_vol_file)
    apix = header.apix

    #GT PDB
    gt_pdb_file =  "../cryofold/AKMD/pdbs/%s_df.pdb"%str(ind+1).zfill(5)
    gt_structure = parser.get_structure("ref", gt_pdb_file)

    for j in range(n_expand):
        i = ind*n_expand +j 
        print("============================== ITER %i ======================================="%i)

        v,s = model.decoder.eval_volume(torch.empty(0).to(device), D-1, None, None,torch.tensor(z[i]).to(device) )

        # Volume FSC
        fsc = get_fsc_curve(v.detach().cpu(), torch.tensor(vol))
        fscauc = np.trapz(np.array(fsc["fsc"]), np.array(fsc["pixres"]))
        fsc_res = get_fsc_thresholds(fsc, apix=apix)
        # create_fsc_plot(fsc, "../cryofold/AKMD/fsc.png", apix)
        # write_mrc("../cryofold/AKMD/test.mrc",v.detach().cpu() )

        # PDB RMSD

        s["final_atom_positions"] = s["final_atom_positions"] @ model.decoder.rot_init + model.decoder.trans_init
        pdb_string = StringIO(struct_to_pdb(tensor_tree_map(lambda x: x.detach().cpu().numpy()[-1], s),"", return_string=True))
        pred_structure = parser.get_structure("mobile",pdb_string)
        rmsd = aligned_rmsd(gt_structure, pred_structure)
        print(rmsd)

        if not "pixres" in outputs : 
            outputs["pixres"] = fsc["pixres"].to_numpy()
        outputs["fsc"].append(fsc["fsc"].to_numpy())
        outputs["fscauc"].append(fscauc)
        outputs["fscres"].append(fsc_res)
        outputs["rmsd"].append(rmsd)

    with open(outfile, "wb") as f:
        pickle.dump(outputs, f)






























import mrcfile
import torch
from flexfold.fsc import fourier_shell_correlation, fsc_auc,fsc_thresh
from cryodrgn.mrcfile import parse_mrc, write_mrc
import time

device = "cuda"

file1 = "../cryofold/AKMD/snr0.01/run/debug_pred.mrc"
file2 = "../cryofold/AKMD/snr0.01/run/debug_gt.mrc"
with mrcfile.open(file1, permissive=True) as mrc:
    vol1 = torch.tensor(mrc.data.copy()  ).to(device)
with mrcfile.open(file2, permissive=True) as mrc:
    vol2 = torch.tensor(mrc.data.copy()  ).to(device)

dt = time.time()
fsc, freqs  = fourier_shell_correlation(vol1, vol2, apix=1.0)
print(time.time()-dt)

file1 = "../cryofold/AKMD/snr0.01/run_cryodrgn/debug_pred.mrc"
file2 = "../cryofold/AKMD/snr0.01/run_cryodrgn/debug_gt.mrc"
with mrcfile.open(file1, permissive=True) as mrc:
    vol1 = torch.tensor(mrc.data.copy()  ).to(device)
with mrcfile.open(file2, permissive=True) as mrc:
    vol2 = torch.tensor(mrc.data.copy()  ).to(device)

dt = time.time()
fsc2, freqs2 = fourier_shell_correlation(vol1, vol2, apix=1.0)
print(time.time()-dt)


auc1 = fsc_auc(fsc, freqs)
auc2 = fsc_auc(fsc2, freqs2)
res_05_1, res_143_1 = fsc_thresh(fsc, freqs)
res_05_2, res_143_2 = fsc_thresh(fsc2, freqs2)

fsc = fsc.cpu().numpy()
fsc2 = fsc2.cpu().numpy()
freqs = freqs.cpu().numpy()
freqs2 = freqs2.cpu().numpy()

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(1,1)
ax.plot(freqs, fsc)
ax.plot(freqs, fsc2)
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.axhline(0.143, c="red")
ax.axhline(0.5, c="green")
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
ax.set_ylim(0,1)
fig.savefig("../cryofold/AKMD/snr0.01/run/test.png")


###################################################"
# "    
def get_atom_coords(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=False) and "CA" in residue:
                    atoms += [a.get_coord() for a in residue]
    return np.array(atoms)
def get_ca_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=False) and "CA" in residue:
                    atoms.append(residue["CA"])
    return atoms
def get_rmsd(arr1, arr2):
    return np.sqrt(np.mean(np.square(np.linalg.norm(arr1-arr2, axis=-1))))

def get_coef(structure):
    coef = []
    for atom in structure.get_atoms():
        coef.append(atomdefs[atom.element][0])
    return np.array(coef)

def vol_from_coords(coords, coef):
    crd = torch.tensor(coords)[None]
    pix_loc, pix_mask = get_voxel_mask(crd, grid_size=100, pixel_size=1.0, n_pix_cutoff=15)
    vol = vol_real_mask(crd=crd, pix_loc=pix_loc, pix_mask=pix_mask, grid_size=100, sigma=1.0, pixel_size=1.0, coef=coef)
    return vol[-1]

from Bio.PDB import PDBParser, Superimposer, is_aa
import numpy as np
from flexfold.core import vol_real_mask, get_voxel_mask,atomdefs
from flexfold.fsc import fourier_shell_correlation, fsc_thresh
from cryodrgn.mrcfile import parse_mrc, write_mrc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
from scipy.ndimage import shift
rundir = "/home/vuillemr/flexfold/data/cryofold/AKMD/snr1/run/debug/debug_%s" %str(20).zfill(6)



gt_vol,_ = parse_mrc(rundir+"_gt.mrc")
pred_vol,_ = parse_mrc(rundir+"_pred.mrc")
pred_vol = shift(pred_vol, np.ones(3)*0.5)
fsc1, freqs = fourier_shell_correlation(torch.tensor(gt_vol), torch.tensor(pred_vol))
res = fsc_thresh(fsc, freqs)
write_mrc(rundir + "_pred_shift.mrc", pred_vol)
print("MRC RESOLUTION %.2f Ang (%.2f Ang)"%(res))

# PARSE
parser = PDBParser(QUIET=True)
gt_structure = parser.get_structure("gt",rundir+"_gt.pdb")
pred_structure = parser.get_structure("pred",rundir+"_pred.pdb")

# GET INITIAL PDBs
gt_coef = get_coef(gt_structure)
pred_coef = get_coef(pred_structure)
gt_atoms = get_atom_coords(gt_structure)
gt_ca = get_ca_atoms(gt_structure)
gt_ca_atoms = np.array([a.get_coord() for a in gt_ca])
pred_atoms = get_atom_coords(pred_structure)
pred_ca = get_ca_atoms(pred_structure)
pred_ca_atoms = np.array([a.get_coord() for a in pred_ca])

gt_vol = vol_from_coords(gt_atoms, gt_coef)
pred_vol = vol_from_coords(pred_atoms, pred_coef)
fsc, freqs = fourier_shell_correlation(gt_vol, pred_vol)
res = fsc_thresh(fsc, freqs)
print("PDB RESOLUTION %.2f Ang (%.2f Ang)"%(res))


translation = gt_atoms.mean(axis=0) - pred_atoms.mean(axis=0)
for atom in pred_structure.get_atoms():
    atom.coord = atom.coord + translation
pred_atoms = get_atom_coords(pred_structure)
pred_ca = get_ca_atoms(pred_structure)
pred_ca_atoms = np.array([a.get_coord() for a in pred_ca])

gt_vol = vol_from_coords(gt_atoms, gt_coef)
pred_vol = vol_from_coords(pred_atoms, pred_coef)
fsc, freqs = fourier_shell_correlation(gt_vol, pred_vol)
res = fsc_thresh(fsc, freqs)
print("PDB RESOLUTION %.2f Ang (%.2f Ang)"%(res))


sup = Superimposer()
sup.set_atoms(gt_ca, pred_ca)  # This aligns atoms2 onto atoms1
rot, tran= sup.rotran
sup.rms
sup.apply(pred_structure.get_atoms())

pred_atoms = get_atom_coords(pred_structure)
pred_ca = get_ca_atoms(pred_structure)
pred_ca_atoms = np.array([a.get_coord() for a in pred_ca])

gt_vol = vol_from_coords(gt_atoms, gt_coef)
pred_vol = vol_from_coords(pred_atoms, pred_coef)
fsc, freqs = fourier_shell_correlation(gt_vol, pred_vol)
res = fsc_thresh(fsc, freqs)
print("PDB RESOLUTION %.2f Ang (%.2f Ang)"%(res))


write_mrc(rundir + "gt_test.mrc",gt_vol)
write_mrc(rundir + "pred_test.mrc", pred_vol)

res1 = res[1]
res5 = res[0]
fig, ax = plt.subplots(1,1)
ax.plot(freqs,fsc)
ax.plot(freqs,fsc1)
def fraction_formatter(x, pos):
    if x == 0:
        return "0"
    return f"1/{x**-1:.1f}"   # reciprocal with 2 decimal places
ax.xaxis.set_major_formatter(FuncFormatter(fraction_formatter))
ax.axhline(0.143, c="red")
ax.axhline(0.5, c="green")
ax.axvline(1/res1, c="red")
ax.axvline(1/res5, c="green")
ax.set_xlabel("Resolution ($1/\AA$)")
ax.set_ylabel("Fourier Shell Correlation")
fig.savefig(rundir+"assert_fsc.png")
plt.close(fig)
