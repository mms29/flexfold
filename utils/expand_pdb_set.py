import os
import glob

in_dir = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/pdbs/000.pdb"
out_dir = "/home/vuillemr/cryofold/cryobench_IgD/IgG-1D/pdbs_expanded_000/"
infiles = glob.glob(in_dir)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

expand = 1000

k=0
for f in infiles:
    for i in range(expand):
        k+=1
        cmd = "ln -s %s %s/%s.pdb"%(f, out_dir, str(k).zfill(6))
        print(cmd)
        os.system(cmd)

