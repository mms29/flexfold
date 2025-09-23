python scripts/run_pretrained_openfold.py  ../cryofold/cryobench_IgD/new_preds     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:1"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/cryobench_IgD/test          --embeddings_output_path data/cryofold/cryobench_IgD/test/test.pt  \
                                              --use_precomputed_alignments data/cryofold/cryobench_IgD/alignments/      --data_random_seed 43 


BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run_target_conv

python -u ./scripts/train_target.py \
    $BASE_DIR/sorted_particles.128.txt  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 10000 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path ~/cryofold/cryobench_IgD/pred2/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose_dummy.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 64 \
    --enc-layers 5 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain real \
    --encode-mode conv \
    --use_lma \
    --pair_stack \
    --frozen_structure_module \
    --target_file ~/cryofold/cryobench_IgD/1HZH.cif \
    --overwrite \
    --multimer 