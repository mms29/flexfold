#Embeddings
python scripts/run_pretrained_openfold.py  data/cryofold/jillsData/sequence/     data/pdb_data/mmcifs/     \
  --uniref90_database_path data/alignment_data/uniref90/uniref90.fasta       --mgnify_database_path data/alignment_data/mgnify/mgy_clusters_2022_05.fa  \
       --pdb_seqres_database_path data/alignment_data/pdb_seqres/pdb_seqres.txt        --uniref30_database_path data/alignment_data/uniref30/UniRef30_2021_03    \
             --uniprot_database_path  data/alignment_data/uniprot/uniprot_trembl.fasta          --jackhmmer_binary_path /home/vuillemr/.conda/envs/flexfold/bin/jackhmmer   \
                    --hhblits_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hhblits          --hmmsearch_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmsearch    \
                          --hmmbuild_binary_path /home/vuillemr/.conda/envs/flexfold/bin/hmmbuild          --kalign_binary_path /home/vuillemr/.conda/envs/flexfold/bin/kalign  \
                                  --config_preset "model_1_multimer_v3"          --model_device "cuda:0"  \
                                  --jax_param_path /home/vuillemr/openfold/openfold/resources/params/params_model_1_multimer_v3.npz\
                                          --output_dir data/cryofold/jillsData/pred2/          --embeddings_output_path data/cryofold/jillsData/pred2/embeddings.pt  \
                                              --use_precomputed_alignments data/cryofold/jillsData/pred2/alignments       --data_random_seed 43 






# Preprocessing
python ~/flexfold/scripts/rln2xmp.py particles_ctf.star particles.xmd
xmipp_transform_filter -i particles.xmd -o particles_filtered.mrcs --save_metadata_stack particles_filtered.xmd --keep_input_columns --fourier low_pass 0.29090
xmipp_image_resize -i particles_filtered.xmd -o particles_downsampled.mrcs --dim 128 --save_metadata_stack particles_downsampled.xmd --keep_input_columns
python ~/flexfold/scripts/rln2xmp.py particles_downsampled.xmd particles_downsampled.star --inverse --optics_group_from particles_ctf.star --add_missing_cols_from particles_ctf.star --pixel_size 1.58 --dimension 128



BASE_DIR="/home/vuillemr/flexfold/data/cryofold/jillsData/particles"
RUN_DIR=$BASE_DIR/run


# Convert metadata
cryodrgn parse_ctf_star $BASE_DIR/particles_downsampled.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_downsampled.star -o $BASE_DIR/particles.pkl

# Backproject for verification
cryodrgn backproject_voxel $BASE_DIR/particles_downsampled.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject

# Initial pose
python ~/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose \
 --from_aligned_pdb  ~/cryofold/jillsData/pred2/predictions/aligned.pdb \
 --alignment_reference  ~/cryofold/jillsData/pred2/predictions/MEK1DDGRA24-ERK2T185V_model_1_multimer_v3_unrelaxed.pdb   \
  --overwrite

python ~/flexfold/scripts/compute_initial_pose.py $BASE_DIR/initial_pose_auto --backproject_path $BASE_DIR/backproject/backproject.mrc\
  --embedding_pdb_path ~/cryofold/jillsData/pred2/predictions/MEK1DDGRA24-ERK2T185V_model_1_multimer_v3_unrelaxed.pdb  \
    --dist_search 5 --grid_size 128 --pixel_size 1.58 --sigma 1.05 --real_space --overwrite

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/jillsData/particles"
RUN_DIR=$BASE_DIR/run_target

python -u ./scripts/train_target.py \
    $BASE_DIR/particles_downsampled.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 10000 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --embedding_path data/cryofold/jillsData/pred2/embeddings_fixed.pt   \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 32 \
    --dec-layers 4 \
    --domain real \
    --encode-mode conv \
    --pair_stack \
    --frozen_structure_module \
    --target_file $BASE_DIR/../MEK1DDGRA_ERK2T185V_ADP_AF3_refined_019.pdb \
    --overwrite \
    --multimer \
      --wd 1e-4\
    --lr 5e-4\
    --warmup 20 


python ~/flexfold//scripts/train.py $BASE_DIR/particles_downsampled.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.58 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path data/cryofold/jillsData/pred2/embeddings_fixed.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 4  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 32 \
    --enc-layers 5 \
    --dec-dim 32 \
    --dec-layers 4 \
    --domain real \
    --encode-mode conv \
    --overwrite\
    --pair_stack \
    --frozen_structure_module \
    --multimer \
    --wd 1e-4 \
    --lr 5e-4 \
    --warmup 100 \
    --debug \

python ~/flexfold//scripts/train.py $BASE_DIR/particles_downsampled.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 1.58 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path data/cryofold/jillsData/pred2/embeddings_fixed.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path ../openfold/openfold/resources/params/params_model_1_multimer_v3.npz \
    --batch-size 2  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 64 \
    --enc-layers 5 \
    --domain real \
    --encode-mode conv \
    --dec-dim 64 \
    --dec-layers 3 \
    --all_atom\
    --overwrite\
    --multimer\
    --frozen_structure_module\
    --pair_stack\
    --debug\

BASE_DIR="/home/vuillemr/flexfold/data/cryofold/jillsData/particles"
RUN_DIR=$BASE_DIR/run_cryodrgn

cryodrgn train_vae $BASE_DIR/particles_downsampled.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --batch-size 1024  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \


cryodrgn analyze  -o $RUN_DIR/analysis $RUN_DIR 64 --pc 2