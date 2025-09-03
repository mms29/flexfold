
import argparse
import logging
import os
import sys
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning import seed_everything
import torch
import wandb
import tqdm
from openfold.config_saxs import model_config_saxs
from openfold.config import model_config
from openfold.utils.logger import PerformanceLoggingCallback
from pytorch_lightning.loggers import TensorBoardLogger

from openfold.utils.saxs_utils import ( SaxsWrapper, SaxsDataModule, resume_ckpt)
from openfold.model.model import AlphaFold

from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import residue_constants, protein

import numpy as np
import time

from openfold.data.data_modules import OpenFoldDataModule, OpenFoldSingleDataset
from openfold.utils.saxs_utils import ( SaxsWrapper, SaxsDataModule, resume_ckpt, output_single_pdb)


def main(args):
    # CONFIG
    config = model_config("finetuning", train=True)
    config.loss.tm.enabled = False
    config.data.common.max_recycling_iters = 3

    # Model
    model = AlphaFold(config)
    model = resume_ckpt("./openfold/resources/openfold_params/finetuning_no_templ_1.pt", model, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    with open("data/alignment_data/alignment_dbs/alignment_db.index", "r") as fp:
        alignment_index = json.load(fp)

    dataset = OpenFoldSingleDataset(
                    data_dir="data/pdb_data/mmcifs",
                    alignment_dir= "data/alignment_data/alignment_dbs/",
                    template_mmcif_dir="data/pdb_data/mmcifs",
                    max_template_date= "2025-06-06",
                    config=config.data,
                    chain_data_cache_path = "data/pdb_data/data_caches/chain_data_cache.json",
                    mode= "eval",
                    alignment_index= alignment_index,
    )


    batch =dataset[dataset.chain_id_to_idx(args.pdbid)]
    # batch =dataset[dataset.chain_id_to_idx("4ake_A")]
    batch = {
        k: torch.as_tensor(v, device=device)
        for k, v in batch.items()
    }
    with torch.no_grad():
        output = model(batch)

    print("pLDDT : %.2f "%torch.mean(output["plddt"]).detach().cpu().numpy())
    ###########################################" CRYOFOLD EMBEDDINGS"
    ###########################################" CRYOFOLD EMBEDDINGS"
    ###########################################" CRYOFOLD EMBEDDINGS"

    del output["msa"]
    del output["sm"]
    del output["final_affine_tensor"]
    del output["num_recycles"]
    del output["lddt_logits"]
    del output["distogram_logits"]
    del output["masked_msa_logits"]
    del output["experimentally_resolved_logits"]
    del output["plddt"]

    del batch["bert_mask"]
    # del batch["alt_chi_angles"]
    del batch["extra_deletion_value"]
    del batch["extra_has_deletion"]
    del batch["extra_msa"]
    del batch["extra_msa_mask"]
    del batch["extra_msa_row_mask"]
    del batch["is_distillation"]
    del batch["msa_feat"]
    del batch["msa_mask"]
    del batch["msa_row_mask"]
    del batch["target_feat"]
    del batch["template_aatype"]
    del batch["template_all_atom_mask"]
    del batch["template_all_atom_positions"]
    del batch["template_alt_torsion_angles_sin_cos"]
    # del batch["template_backbone_rigid_mask"]
    # del batch["template_backbone_rigid_tensor"]
    del batch["template_mask"]
    del batch["template_pseudo_beta"]
    del batch["template_pseudo_beta_mask"]
    del batch["template_sum_probs"]
    del batch["template_torsion_angles_mask"]
    # del batch["template_torsion_angles_sin_cos "]
    del batch["true_msa"]
    del batch["batch_idx"]

    embeddings = tensor_tree_map(lambda x: x[..., -1],batch)
    embeddings.update(output)

    prefix =  "%s/%s_embeddings"%(args.outdir, args.pdbid)
    torch.save(embeddings,prefix + ".pt")
    print("Successfully written %s.pt"%prefix)


    batch_out = tensor_tree_map(lambda x: np.array(x.detach().cpu())[..., -1],batch)
    output = tensor_tree_map(lambda x: np.array(x.detach().cpu()), output)
    output_single_pdb(output["final_atom_positions"], batch_out["aatype"], output["final_atom_mask"],  prefix+".pdb")
    print("Successfully written %s.pdb"%prefix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( "pdbid", type=str,help="")
    parser.add_argument( "outdir", type=str,help="")

    args = parser.parse_args()
    main(args)