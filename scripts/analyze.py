"""Visualize latent space and generate volumes using a trained cryoDRGN model.

Example usage
-------------
$ cryodrgn analyze 003_abinit-het/ 49

# It is necessary to invert handedness for some datasets
$ cryodrgn analyze 003_abinit-het/ 99 --invert

# Avoid running more computationally expensive analyses
$ cryodrgn analyze 003_abinit-het/ 99 --skip-umap --skip-vol

"""
import argparse
import os
import os.path
import shutil
from datetime import datetime as dt
import logging
import nbformat
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cryodrgn
from cryodrgn import utils, config
from cryodrgn.commands import backproject_voxel
from flexfold import analysis
import pandas as pd

logger = logging.getLogger(__name__)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "workdir", type=os.path.abspath, help="Directory with cryoDRGN results"
    )
    parser.add_argument(
        "epoch",
        type=int,
        help="Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)",
    )
    parser.add_argument("--device", type=int, help="Optionally specify CUDA device")
    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory for analysis results (default: [workdir]/analyze.[epoch])",
    )
    parser.add_argument(
        "--skip-vol", action="store_true", help="Skip generation of volumes"
    )
    parser.add_argument("--skip-umap", action="store_true", help="Skip running UMAP")

    group = parser.add_argument_group("Extra arguments for volume generation")
    group.add_argument(
        "--Apix",
        type=float,
        help="Pixel size to add to .mrc header (default is to infer from ctf.pkl file else 1)",
    )
    group.add_argument(
        "--flip", action="store_true", help="Flip handedness of output volumes"
    )
    group.add_argument(
        "--invert", action="store_true", help="Invert contrast of output volumes"
    )
    group.add_argument(
        "-d",
        "--downsample",
        type=int,
        help="Downsample volumes to this box size (pixels)",
    )
    group.add_argument(
        "--pc",
        type=int,
        default=2,
        help="Number of principal component traversals to generate (default: %(default)s)",
    )
    group.add_argument(
        "--ksample",
        type=int,
        default=20,
        help="Number of kmeans samples to generate (default: %(default)s)",
    )
    group.add_argument(
        "--vol-start-index",
        type=int,
        default=0,
        help="Default value of start index for volume generation (default: %(default)s)",
    )
    group.add_argument(
        "--no_volume", action="store_true", help="TODO"
    )
    group.add_argument(
        "--trajectory", action="store_true", help="TODO"
    )
    group.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="TODO",
    )
    group.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="TODO",
    )
    group.add_argument(
        "--backproject", action="store_true", help="TODO"
    )
    group.add_argument(
        "--particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
        default=None
    )
    group.add_argument(
        "--poses", type=os.path.abspath, help="Image poses (.pkl)",default=None
    )
    group.add_argument(
        "--ctf",
        metavar="pkl",
        type=os.path.abspath,
        help="CTF parameters (.pkl) for phase flipping images",default=None
    )
    group.add_argument(
        "--no_fsc", action="store_true", help="TODO"
    )
    return parser


def analyze_z1(z, outdir, vg):
    """Plotting and volume generation for 1D z"""
    assert z.shape[1] == 1
    z = z.reshape(-1)
    N = len(z)

    plt.figure(1)
    plt.scatter(np.arange(N), z, alpha=0.1, s=2)
    plt.xlabel("particle")
    plt.ylabel("z")
    plt.savefig(f"{outdir}/z.png")

    plt.figure(2)
    sns.distplot(z)
    plt.xlabel("z")
    plt.savefig(f"{outdir}/z_hist.png")

    ztraj = np.percentile(z, np.linspace(5, 95, 10))
    vg.gen_volumes(outdir, ztraj)


def run_backproject(particles, poses, ctf, outdir, indices, no_fsc):

    args = [particles, "--poses", poses, "--ctf", ctf,"-o", outdir, "--ind", indices]
    if no_fsc : 
        args+= ["--no-half-maps", "--no-fsc-vals"]
    parser = argparse.ArgumentParser()
    backproject_voxel.add_args(parser)
    return backproject_voxel.main(parser.parse_args(args))

def backproject_closest(closest, outdir, particles, poses, ctf, no_fsc):
    indices = np.sort(np.unique(closest))

    for i in indices:
        ind_path = f"{outdir}/{i}_ind.pkl"
        utils.save_pkl(np.where(i==closest)[0],ind_path )
        run_backproject(
            particles=particles, poses=poses, ctf=ctf, indices=ind_path, no_fsc=no_fsc,
            outdir = f"{outdir}/{i}_backproject"
        )
        # outfile = f"{outdir}/{i}_backproject/backproject.mrc"
        # outlink = f"{outdir}/{i}_backproject.mrc"
        # os.system("ln -s %s %s;" %(outfile, outlink))




def analyze_zN(
    z, outdir, vg, workdir, epoch, skip_umap=False, num_pcs=2, num_ksamples=20, size= 1.0, alpha=0.1, dpi=300, 
    backproject_args=None,
):
    zdim = z.shape[1]

    def get_closest(z_pc, z):
        return  np.argmin(
                    np.linalg.norm(
                        z_pc[None, :, :] - z[:, None, :], 
                        axis=-1
                    ), 
                    axis=-1
        )

    # Principal component analysis
    logger.info("Performing principal component analysis...")
    pc, pca = analysis.run_pca(z)
    logger.info("Generating volumes...")

    for i in range(num_pcs):
        start, end = np.percentile(pc[:, i], (5, 95))
        z_pc = analysis.get_pc_traj(pca, z.shape[1], 10, i + 1, start, end)
        closest = get_closest(z_pc, z)

        vg.gen_volumes(f"{outdir}/pc{i+1}", z_pc)

        if backproject_args is not None:
            backproject_closest(closest=closest, outdir = f"{outdir}/pc{i+1}/", **backproject_args)

    # kmeans clustering
    logger.info("K-means clustering...")
    K = num_ksamples
    kmeans_labels, centers = analysis.cluster_kmeans(z, K)
    centers, centers_ind = analysis.get_nearest_point(z, centers)
    if not os.path.exists(f"{outdir}/kmeans{K}"):
        os.mkdir(f"{outdir}/kmeans{K}")
    utils.save_pkl(kmeans_labels, f"{outdir}/kmeans{K}/labels.pkl")
    np.savetxt(f"{outdir}/kmeans{K}/centers.txt", centers)
    np.savetxt(f"{outdir}/kmeans{K}/centers_ind.txt", centers_ind, fmt="%d")
    logger.info("Generating volumes...")
    vg.gen_volumes(f"{outdir}/kmeans{K}", centers)

    # UMAP -- slow step
    umap_emb = None
    if zdim > 2 and not skip_umap:
        logger.info("Running UMAP...")
        umap_emb = analysis.run_umap(z)
        utils.save_pkl(umap_emb, f"{outdir}/umap.pkl")

    # Make some plots
    logger.info("Generating plots...")

    # Plot learning curve
    loss = analysis.parse_loss(f"{workdir}/metrics.csv")
    plt.figure(figsize=(4, 4))
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axvline(x=epoch, linestyle="--", color="black", label=f"Epoch {epoch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/learning_curve_epoch{epoch}.png", dpi=dpi)
    plt.close()

    def plt_pc_labels(x=0, y=1):
        plt.xlabel(f"PC{x+1} ({pca.explained_variance_ratio_[x]:.2f})")
        plt.ylabel(f"PC{y+1} ({pca.explained_variance_ratio_[y]:.2f})")

    def plt_pc_labels_jointplot(g, x=0, y=1):
        g.ax_joint.set_xlabel(f"PC{x+1} ({pca.explained_variance_ratio_[x]:.2f})")
        g.ax_joint.set_ylabel(f"PC{y+1} ({pca.explained_variance_ratio_[y]:.2f})")

    def plt_umap_labels():
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

    def plt_umap_labels_jointplot(g):
        g.ax_joint.set_xlabel("UMAP1")
        g.ax_joint.set_ylabel("UMAP2")

    # PCA -- Style 1 -- Scatter
    plt.figure(figsize=(4, 4))
    plt.scatter(pc[:, 0], pc[:, 1], alpha=alpha, s=size, rasterized=True)
    plt_pc_labels()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{outdir}/z_pca.png", dpi=dpi)
    plt.close()

    # PCA -- Style 2 -- Scatter, with marginals
    g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], alpha=alpha, s=size, rasterized=True, height=4)
    plt_pc_labels_jointplot(g)
    g.ax_joint.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{outdir}/z_pca_marginals.png", dpi=dpi)
    plt.close()

    # PCA -- Style 3 -- Hexbin
    try:
        g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], height=4, kind="hex")
        plt_pc_labels_jointplot(g)
        g.ax_joint.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f"{outdir}/z_pca_hexbin.png", dpi=dpi)
        plt.close()
    except ZeroDivisionError:
        print("Data too small to produce hexbins!")

    if umap_emb is not None:
        # Style 1 -- Scatter
        plt.figure(figsize=(4, 4))
        plt.scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=alpha, s=size, rasterized=True)
        plt_umap_labels()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{outdir}/umap.png", dpi=dpi)
        plt.close()

        # Style 2 -- Scatter with marginal distributions
        try:
            g = sns.jointplot(
                x=umap_emb[:, 0],
                y=umap_emb[:, 1],
                alpha=alpha,
                s=size,
                rasterized=True,
                height=4,
            )
            plt_umap_labels_jointplot(g)
            g.ax_joint.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(f"{outdir}/umap_marginals.png", dpi=dpi)
            plt.close()
        except ZeroDivisionError:
            logger.warning("Data too small for marginal distribution scatterplots!")

        # Style 3 -- Hexbin / heatmap
        try:
            g = sns.jointplot(x=umap_emb[:, 0], y=umap_emb[:, 1], kind="hex", height=4)
            plt_umap_labels_jointplot(g)
            g.ax_joint.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(f"{outdir}/umap_hexbin.png", dpi=dpi)
            plt.close()
        except ZeroDivisionError:
            logger.warning("Data too small to generate UMAP hexbins!")

    # Plot kmeans sample points
    colors = analysis._get_chimerax_colors(K)
    analysis.scatter_annotate(
        pc[:, 0],
        pc[:, 1],
        centers_ind=centers_ind,
        annotate=True,
        colors=colors,
    )
    plt_pc_labels()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{outdir}/kmeans{K}/z_pca.png", dpi=dpi)
    plt.close()

    try:
        g = analysis.scatter_annotate_hex(
            pc[:, 0],
            pc[:, 1],
            centers_ind=centers_ind,
            annotate=True,
            colors=colors,
        )
    except ZeroDivisionError:
        logger.warning("Data too small to generate PCA annotated hexes!")

    plt_pc_labels_jointplot(g)
    g.ax_joint.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{outdir}/kmeans{K}/z_pca_hex.png", dpi=dpi)
    plt.close()

    if umap_emb is not None:
        analysis.scatter_annotate(
            umap_emb[:, 0],
            umap_emb[:, 1],
            centers_ind=centers_ind,
            annotate=True,
            colors=colors,
        )
        plt_umap_labels()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{outdir}/kmeans{K}/umap.png", dpi=dpi)
        plt.close()

        try:
            g = analysis.scatter_annotate_hex(
                umap_emb[:, 0],
                umap_emb[:, 1],
                centers_ind=centers_ind,
                annotate=True,
                colors=colors,
            )
            plt_umap_labels_jointplot(g)
            g.ax_joint.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(f"{outdir}/kmeans{K}/umap_hex.png", dpi=dpi)
            plt.close()
        except ZeroDivisionError:
            logger.warning("Data too small to generate UMAP annotated hexes!")

    # Plot PC trajectories
    for i in range(num_pcs):
        start, end = np.percentile(pc[:, i], (5, 95))
        z_pc = analysis.get_pc_traj(pca, z.shape[1], 10, i + 1, start, end)
        closest = get_closest(z_pc, z)

        if umap_emb is not None:
            # UMAP, colored by PCX
            analysis.scatter_color(
                umap_emb[:, 0],
                umap_emb[:, 1],
                pc[:, i],
                label=f"PC{i+1}",
                s=size,
                alpha=alpha
            )
            plt_umap_labels()
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{outdir}/pc{i+1}/umap.png", dpi=dpi)
            plt.close()

            # UMAP, with PC traversal
            z_pc_on_data, pc_ind = analysis.get_nearest_point(z, z_pc)
            dists = ((z_pc_on_data - z_pc) ** 2).sum(axis=1) ** 0.5
            if np.any(dists > 2):
                logger.warning(
                    f"Warning: PC{i+1} point locations in UMAP plot may be inaccurate"
                )
            plt.figure(figsize=(4, 4))
            plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], alpha=alpha, s=size, rasterized=True, c=closest, cmap="rocket"
            )
            plt.scatter(
                umap_emb[pc_ind, 0],
                umap_emb[pc_ind, 1],
                c="cornflowerblue",
                edgecolor="black",
            )
            plt_umap_labels()
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{outdir}/pc{i+1}/umap_traversal.png", dpi=dpi)
            plt.close()

            # UMAP, with PC traversal, connected
            plt.figure(figsize=(4, 4))
            plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], alpha=alpha, s=size, rasterized=True, c=closest, cmap="rocket"
            )
            plt.plot(umap_emb[pc_ind, 0], umap_emb[pc_ind, 1], "--", c="k")
            plt.scatter(
                umap_emb[pc_ind, 0],
                umap_emb[pc_ind, 1],
                c="cornflowerblue",
                edgecolor="black",
            )
            plt_umap_labels()
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{outdir}/pc{i+1}/umap_traversal_connected.png", dpi=dpi)
            plt.close()

        # 10 points, from 5th to 95th percentile of PC1 values
        print(closest)
        t = np.linspace(start, end, 10, endpoint=True)
        plt.figure(figsize=(4, 4))
        if i > 0 and i == num_pcs - 1:
            plt.scatter(pc[:, i - 1], pc[:, i], alpha=alpha, s=size, rasterized=True, c=closest, cmap="rocket")
            plt.scatter(np.zeros(10), t, c="cornflowerblue", edgecolor="white")
            plt_pc_labels(i - 1, i)
        else:
            plt.scatter(pc[:, i], pc[:, i + 1], alpha=alpha, s=size, rasterized=True, c=closest, cmap="rocket")
            plt.scatter(t, np.zeros(10), c="cornflowerblue", edgecolor="white")
            plt_pc_labels(i, i + 1)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{outdir}/pc{i+1}/pca_traversal.png", dpi=dpi)
        plt.close()

        if i > 0 and i == num_pcs - 1:
            g = sns.jointplot(
                x=pc[:, i - 1], y=pc[:, i], alpha=alpha, s=size, rasterized=True, height=4, hue=closest
            )
            g.ax_joint.scatter(np.zeros(10), t, c="cornflowerblue", edgecolor="white")
            plt_pc_labels_jointplot(g, i - 1, i)
        else:
            g = sns.jointplot(
                x=pc[:, i], y=pc[:, i + 1], alpha=alpha, s=size, rasterized=True, height=4, hue=closest
            )
            g.ax_joint.scatter(t, np.zeros(10), c="cornflowerblue", edgecolor="white")
            plt_pc_labels_jointplot(g)
        g.ax_joint.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f"{outdir}/pc{i+1}/pca_traversal_hex.png", dpi=dpi)
        plt.close()


class VolumeGenerator:
    """Helper class to call analysis.gen_volumes"""

    def __init__(self, weights, config, vol_args={}, skip_vol=False):
        self.weights = weights
        self.config = config
        self.vol_args = vol_args
        self.skip_vol = skip_vol

    def gen_volumes(self, outdir, z_values):
        if self.skip_vol:
            return

        os.makedirs(outdir, exist_ok=True)
        zfile = f"{outdir}/z_values.txt"
        np.savetxt(zfile, z_values)
        analysis.gen_volumes(self.weights, self.config, zfile, outdir, **self.vol_args)


def loss_plot(infile, oufile, w = 10):
    
    movavg = lambda arr: np.convolve(
        np.nan_to_num(arr), np.ones(w), 'valid'
    ) / np.convolve(~np.isnan(arr), np.ones(w), 'valid')

    losses=["data_loss_step", "chi_loss_step", "viol_loss_step","center_loss_step", "kld_step", "loss_step"]
    col = "tab:blue"
    nrows = 2
    ncols=3
    fig, ax = plt.subplots(nrows,ncols, figsize=(10,5), layout="constrained")
    for x in range(nrows):
        for y in range(ncols):
            ii = x *ncols + y
            if ii>=len(losses):
                break

            metrics =pd.read_csv(infile)
            ax[x,y].plot(metrics["step"], metrics[losses[ii]], alpha=0.5, c=col)
            ax[x,y].plot(metrics["step"][:-(w-1)]*(len(metrics["step"])/(len(metrics["step"])-w)), movavg(metrics[losses[ii]]), label = "run", c=col)
            ax[x,y].set_xlabel("step")
            ax[x,y].set_ylabel(losses[ii])
    ax[-1,-1].legend()
    fig.savefig(oufile, dpi=300)

    plt.close(fig)



def main(args: argparse.Namespace) -> None:
    matplotlib.use("Agg")  # non-interactive backend
    t1 = dt.now()
    E = args.epoch
    workdir = args.workdir
    epoch = args.epoch

    zfile = f"{workdir}/z.{E}.pkl"
    weights = f"{workdir}/weights.{E}.pkl"
    cfg = (
        f"{workdir}/config.yaml"
        if os.path.exists(f"{workdir}/config.yaml")
        else f"{workdir}/config.pkl"
    )

    configs = config.load(cfg)
    outdir = f"{workdir}/analyze.{E}"

    if args.Apix:
        use_apix = args.Apix

    # find A/px from CTF if not given
    else:
        if configs["dataset_args"]["ctf"]:
            ctf_params = utils.load_pkl(configs["dataset_args"]["ctf"])
            orig_apixs = set(ctf_params[:, 1])

            # TODO: add support for multiple optics groups
            if len(orig_apixs) > 1:
                use_apix = 1.0
                logger.info(
                    "cannot find unique A/px in CTF parameters, "
                    "defaulting to A/px=1.0"
                )

            else:
                orig_apix = tuple(orig_apixs)[0]
                orig_sizes = set(ctf_params[:, 0])
                orig_size = tuple(orig_sizes)[0]

                if len(orig_sizes) > 1:
                    logger.info(
                        "cannot find unique original box size in CTF "
                        f"parameters, defaulting to first found: {orig_size}"
                    )

                cur_size = configs["lattice_args"]["D"] - 1
                use_apix = round(orig_apix * orig_size / cur_size, 6)
                logger.info(f"using A/px={use_apix} as per CTF parameters")

        else:
            use_apix = 1.0
            logger.info("Cannot find A/px in CTF parameters, defaulting to A/px=1.0")

    if E == -1:
        zfile = f"{workdir}/z.pkl"
        weights = f"{workdir}/weights.pkl"
        outdir = f"{workdir}/analyze"

    if args.outdir:
        outdir = args.outdir
    logger.info(f"Saving results to {outdir}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    loss_plot(f"{workdir}/metrics.csv", f"{outdir}/losses.png")

    z = utils.load_pkl(zfile)
    zdim = z.shape[1]

    vol_args = dict(
        Apix=use_apix,
        downsample=args.downsample,
        flip=args.flip,
        device=args.device,
        invert=args.invert,
        vol_start_index=args.vol_start_index,
        no_volume=args.no_volume
    )
    vg = VolumeGenerator(weights, cfg, vol_args, skip_vol=args.skip_vol)

    if args.trajectory : 
        vg.vol_args["no_volume"] = True
        vg.gen_volumes(f"{outdir}/traj", z)
        vg.vol_args["no_volume"] = args.no_volume

    if args.backproject:
        backproject_args = {
            "particles":args.particles,
            "poses":args.poses,
            "ctf":args.ctf,
            "no_fsc":args.no_fsc
        }
    else:
        backproject_args = None

    if zdim == 1:
        analyze_z1(z, outdir, vg)
    else:
        analyze_zN(
            z,
            outdir,
            vg,
            workdir,
            epoch,
            skip_umap=args.skip_umap,
            num_pcs=args.pc,
            num_ksamples=args.ksample,
            alpha=args.alpha,
            size=args.size,
            backproject_args=backproject_args
        )

    # create demonstration Jupyter notebooks from templates if they don't already exist
    cfg = config.load(cfg)
    ipynbs = ["cryoDRGN_figures"]
    if cfg["model_args"]["encode_mode"] == "tilt":
        ipynbs += ["cryoDRGN_ET_viz"]
    else:
        ipynbs += ["cryoDRGN_viz", "cryoDRGN_filtering"]

    for ipynb in ipynbs:
        nb_outfile = os.path.join(outdir, f"{ipynb}.ipynb")

        if not os.path.exists(nb_outfile):
            logger.info(f"Creating demo Jupyter notebook {nb_outfile}...")
            nb_infile = os.path.join(
                cryodrgn._ROOT, "templates", f"{ipynb}_template.ipynb"
            )
            shutil.copyfile(nb_infile, nb_outfile)
        else:
            logger.info(f"{nb_outfile} already exists. Skipping")

        # lazily look at the beginning of the notebook for the epoch number to update
        with open(nb_outfile, "r") as f:
            filter_ntbook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        for cell in filter_ntbook["cells"]:
            cell["source"] = cell["source"].replace("EPOCH = None", f"EPOCH = {epoch}")
            cell["source"] = cell["source"].replace(
                "KMEANS = None", f"KMEANS = {args.ksample}"
            )

        with open(nb_outfile, "w") as f:
            nbformat.write(filter_ntbook, f)

    logger.info(f"Finished in {dt.now() - t1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)