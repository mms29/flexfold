
import torch
import argparse
from flexfold.models import struct_to_crd
from cryodrgn.mrcfile import write_mrc, parse_mrc
from flexfold.core import vol_real, vol_ft, register_crd_to_vol,matrix2euler
import torch

from cryodrgn import fft
import sys
import os
from Bio.PDB import PDBParser, Superimposer, is_aa

def output_single_pdb(all_atom_positions, aatype, all_atom_mask, file):

    chain_index = np.zeros_like(aatype)
    b_factors = np.zeros_like(all_atom_mask)
    residue_index = np.arange(len(aatype))+1

    pdb_elem = protein.Protein(
        aatype=aatype,
        atom_positions=all_atom_positions,
        atom_mask=all_atom_mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index,
        remark="",
        parents=None,
        parents_chain_index=None,
    )
    outstring = protein.to_pdb(pdb_elem)
    with open(file, 'w') as fp:
        fp.write(outstring)

# Select atoms (e.g., CA atoms for proteins)
def get_ca_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and "CA" in residue:
                    atoms.append(residue["CA"])
    return atoms

def main(args):

    outfile = args.outdir + ".pt"
    if os.path.isfile(outfile) and not args.overwrite:
        print("Path exists : %s "%(outfile))
        print("Exiting ...")
        sys.exit()



    if args.from_aligned_pdb is not None:
        parser = PDBParser(QUIET=True)
        structure1 = parser.get_structure("ref", args.alignment_reference)
        structure2 = parser.get_structure("mobile", args.from_aligned_pdb)
        atoms1 = get_ca_atoms(structure1)
        atoms2 = get_ca_atoms(structure2)

        # Superimpose and calculate RMSD
        sup = Superimposer()
        sup.set_atoms(atoms2, atoms1)  # This aligns atoms2 onto atoms1
        R_final, shift_final = sup.rotran
        angle_final = matrix2euler(R_final)

        #to tensors
        R_final = torch.tensor(R_final)
        shift_final = torch.tensor(shift_final)
        angle_final = torch.tensor(angle_final)
    else:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        embeddings = torch.load(args.embedding_path)
        crd = embeddings["final_atom_positions"]
        if args.dist_search == -1:
            R_final = torch.eye(3, device=crd.device, dtype=crd.dtype)
            shift_final = torch.zeros(3, device=crd.device, dtype=crd.dtype)
        else:
            crd = struct_to_crd(embeddings, ca=True)
            vol_ext, header = parse_mrc(args.backproject_path)
            vol_ext = torch.tensor(vol_ext).to(device)
            vol_ext = fft.fftn_center(vol_ext)
            assert(vol_ext.shape[-1] == vol_ext.shape[-2] == vol_ext.shape[-3])
            angle_final, R_final, shift_final = register_crd_to_vol(
                vol = vol_ext,
                crd=crd, 
                grid_size=vol_ext.shape[-1], 
                sigma=args.sigma, 
                pixel_size=args.pixel_size, 
                dist_search = args.dist_search,
                real_space = args.real_space
            )

    torch.save({"R": R_final, "T":shift_final}, outfile)

    print("Translation (Angstroms) : %s"%str(shift_final))
    print("Angle (degrees) : %s"%str(angle_final))
    print("Rotation matrix  : %s"%str(R_final))


    print("Successfully written %s"%( args.outdir+".pt"))

    if not args.from_aligned_pdb:
        aligned_crd = embeddings["final_atom_positions"] @ R_final + shift_final
        output_single_pdb(aligned_crd.cpu().numpy(), embeddings["aatype"].cpu().numpy(), embeddings["final_atom_mask"].cpu().numpy(),  args.outdir+".pdb")
        print("Successfully written %s"%( args.outdir+".pdb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( "outdir", type=str,help="")

    parser.add_argument( "--backproject_path", type=str,help="")
    parser.add_argument( "--embedding_path", type=str,help="")
    parser.add_argument( "--dist_search", type=float,help="")
    parser.add_argument( "--grid_size", type=int,help="")
    parser.add_argument( "--pixel_size", type=float,help="")
    parser.add_argument( "--sigma", type=float,help="")
    parser.add_argument( "--overwrite",  action="store_true")
    parser.add_argument( "--real_space",  action="store_true")

    parser.add_argument( "--from_aligned_pdb", type=str,help="", default=None)
    parser.add_argument( "--alignment_reference", type=str,help="", default=None)

    args = parser.parse_args()
    main(args)