
python scripts/run_pretrained_openfold.py  ../cryofold/spike-md/sequence     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/mamba_env/envs/openfold_env/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:1"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir ../cryofold/spike-md   --long_sequence_inference --embeddings_output_path ../cryofold/spike-md/embeddings.pt  \
                                            --data_random_seed 42  \
                                              --use_precomputed_alignments data/cryofold/spike-md/alignments


BASE_DIR="../cryofold/spike-md"
RUN_DIR=$BASE_DIR/run_test

cryodrgn parse_ctf_star $BASE_DIR/images/snr0.1/particles.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/images/snr0.1/particles.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/images/snr0.1/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject


python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose \
 --from_aligned_pdb data/cryofold/spike-md/aligned.pdb \
 --alignment_reference data/cryofold/spike-md/test.pdb  --overwrite


python -u ./scripts/train.py \
  $BASE_DIR/images/snr0.1/particles.mrcs  \
  --poses  $BASE_DIR/particles.pkl\
  --ctf $BASE_DIR/ctf.pkl \
  -n 100 \
  -o $RUN_DIR \
  --pixel_size 1.5 \
  --sigma 2.5\
  --quality_ratio 5.0 \
  --embedding_path ~/cryofold/spike-md/embeddings_crop_mask.pt  \
  --initial_pose_path $BASE_DIR/initial_pose.pt \
  --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
  --batch-size 1  \
  --num-workers 0 \
  --zdim 2  \
  --enc-dim 16 \
  --enc-layers 6 \
  --dec-dim 32 \
  --dec-layers 1 \
  --encode-mode conv\
  --pair_stack\
  --frozen_structure_module\
    --wd 1e-4\
  --lr 5e-4\
  --warmup 100 \
  --domain real \
  --overwrite \
  --multimer \
  --debug \
  --mpi_plugin\
  --num_nodes 2\
  --devices 4\
  # --encode-mode conv\
  # --domain real \
  # --load ~/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/weights.4000.pkl \