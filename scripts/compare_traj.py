import numpy as np
import glob
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, Superimposer, is_aa
import argparse



# Select atoms (e.g., CA atoms for proteins)
def get_ca_atoms(structure):
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and "CA" in residue:
                    atoms.append(residue["CA"])
    return atoms

def aligned_rmsd(structure1, structure2):
    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)

    # Superimpose and calculate RMSD
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)  # This aligns atoms2 onto atoms1
    rmsd = sup.rms
    return rmsd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( "first_set", type=str,help="")
    parser.add_argument( "second_set", type=str,help="")
    parser.add_argument( "output", type=str,help="")

    args = parser.parse_args()
    
    first_set_files = sorted(glob.glob(args.first_set))
    second_set_files = sorted(glob.glob(args.second_set))

    if len(first_set_files) != len(second_set_files):
        raise RuntimeError("Number of files differs!")
    
    rmsds = []
    parser = PDBParser(QUIET=True)
    for i in range(len(first_set_files)):
        f1 = first_set_files[i]
        f2 = second_set_files[i]

        print("Aligning %s -> %s "%(f1,f2))
        structure1 = parser.get_structure("ref", f1)
        structure2 = parser.get_structure("mobile", f2)
        rmsd = aligned_rmsd(structure1, structure2)
        rmsds.append(rmsd)

    print(f"RMSD mean = {np.mean(rmsds):.3f} Å")
    print(f"RMSD std = {np.std(rmsds):.3f} Å")

    np.savetxt(args.output, np.array([np.mean(rmsds), np.std(rmsds)]))


