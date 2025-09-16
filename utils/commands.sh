
#******************************
BASE_DIR="../cryofold/particlesSNR1.0"
RUN_DIR=$BASE_DIR/run_test
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 50 \
    -o $RUN_DIR \
    --pixel_size 2.2 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 16  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain fourier \
    --all_atom \
    --real_space \
    --center_loss_weight 0.01


cryodrgn analyze -o $RUN_DIR/analysis $RUN_DIR 49 --pc 2

python animate_trajectory.py $RUN_DIR/analysis/animate_pc1.cxc --pc_ind 1 --nres 424 --gaussian_sigma 1.5 --grid_size 128
python animate_trajectory.py $RUN_DIR/analysis/animate_pc2.cxc --pc_ind 2 --nres 424 --gaussian_sigma 1.5 --grid_size 128
python animate_trajectory.py $RUN_DIR/analysis/animate_pc3.cxc --pc_ind 3 --nres 424 --gaussian_sigma 1.5 --grid_size 128
python animate_trajectory.py $RUN_DIR/analysis/animate_pc4.cxc --pc_ind 4 --nres 424 --gaussian_sigma 1.5 --grid_size 128



#******************************
BASE_DIR="../cryofold/particlesSNR1.0"
RUN_DIR=$BASE_DIR/run_fourier_inf_baseline
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 2.2 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain fourier \
    --all_atom \
    --real_space \
    --center_loss_weight 0.01\
    --overwrite
python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 --trajectory
python ./scripts/compare_traj.py "$RUN_DIR/analysis/traj/*pdb" "$BASE_DIR/gt_pdbs/*pdb" $RUN_DIR/analysis/stats.txt
#******************************
BASE_DIR="../cryofold/particlesSNR1.0"
RUN_DIR=$BASE_DIR/bench_real
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 2.2 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain fourier \
    --all_atom \
    --real_space \
    --overwrite
python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 --trajectory
python ./scripts/compare_traj.py "$RUN_DIR/analysis/traj/*pdb" "$BASE_DIR/gt_pdbs/*pdb" $RUN_DIR/analysis/stats.txt
#******************************************************
BASE_DIR="../cryofold/particlesSNR1.0"
RUN_DIR=$BASE_DIR/bench_real_direct
python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 2.2 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain fourier \
    --all_atom \
    --overwrite
python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 --trajectory
python ./scripts/compare_traj.py "$RUN_DIR/analysis/traj/*pdb" "$BASE_DIR/gt_pdbs/*pdb" $RUN_DIR/analysis/stats.txt


BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run_test

cryodrgn parse_ctf_star $BASE_DIR/particles_*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/sorted_particles.128.txt  --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject

cryodrgn parse_ctf_star $BASE_DIR/snr0.01_099.star -o $BASE_DIR/ctf_099.pkl
cryodrgn parse_pose_star $BASE_DIR/snr0.01_099.star -o $BASE_DIR/particles_099.pkl
cryodrgn backproject_voxel $BASE_DIR/099_particles_128.mrcs  --poses $BASE_DIR/particles_099.pkl --ctf $BASE_DIR/ctf_099.pkl -o $BASE_DIR/backproject_099


python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose \
 --from_aligned_pdb data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/backproject/aligned_target_fit.pdb \
 --alignment_reference data//cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/fit.4000.pdb  --overwrite
python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose_auto --backproject_path data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/backproject/backproject.mrc\
  --embedding_pdb_path data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/fit.4000.pdb \
    --dist_search 5 --grid_size 100 --pixel_size 3.0 --sigma 1.05 --real_space --overwrite
python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose_dummy --backproject_path data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/backproject/backproject.mrc\
  --embedding_pdb_path data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/fit.6000.pdb \
    --dist_search -1 --grid_size 100 --pixel_size 3.0 --sigma 1.05 --real_space --overwrite

nohup \
python ./scripts/train.py $BASE_DIR/sorted_particles.128.txt  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ~/cryofold/cryobench_IgD/pred2/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz \
    --batch-size 2  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain hartley \
    --load ~/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/weights.4000.pkl \
    --overwrite \
    --multimer \
    --center_loss_weight 0.001    \
> log.txt &\
disown


python -u ./scripts/train.py \
    $BASE_DIR/000_particles_128.mrcs \
    --poses  $BASE_DIR/particles_000.pkl\
    --ctf $BASE_DIR/ctf_000.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ~/cryofold/cryobench_IgD/pred2/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose_auto.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 256 \
    --dec-layers 3 \
    --domain fourier \
    --load ~/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/weights.4000.pkl \
    --overwrite \
    --multimer \
    --center_loss_weight 0.001  \
    --num_nodes 1\
    --devices 1\
    --debug

cryodrgn parse_ctf_star $BASE_DIR/snr0.01_000.star -o $BASE_DIR/ctf_000.pkl
cryodrgn parse_pose_star $BASE_DIR/snr0.01_000.star -o $BASE_DIR/particles_000.pkl
cryodrgn backproject_voxel $BASE_DIR/000_particles_128.mrcs  --poses $BASE_DIR/particles_000.pkl --ctf $BASE_DIR/ctf_000.pkl -o $BASE_DIR/backproject_000

python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose_000 \
 --from_aligned_pdb ~/cryofold/cryobench_IgD/pred2/aligned.pdb --alignment_reference ~/cryofold/cryobench_IgD/pred2/unrelaxed.pdb  --overwrite


python ./scripts/analyze.py -o $RUN_DIR/analysis $RUN_DIR 0 --pc 2

BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run_target_large

python -u ./scripts/train_target.py \
    $BASE_DIR/sorted_particles.128.txt  \
    --poses  $BASE_DIR/particles.pkl\
    --ctf $BASE_DIR/ctf.pkl \
    -n 10000 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05\
    --quality_ratio 5.0 \
    --af_decoder \
    --embedding_path ~/cryofold/cryobench_IgD/pred2/embeddings.pt  \
    --initial_pose_path $BASE_DIR/initial_pose_dummy.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 8  \
    --enc-dim 1024 \
    --enc-layers 3 \
    --dec-dim 1024 \
    --dec-layers 3 \
    --domain fourier \
    --target_file ~/cryofold/cryobench_IgD/1HZH.cif \
    --overwrite \
    --multimer \
    --center_loss_weight 0.01  

################################################################################# JILLL

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

















#####################################" IgD REMI"

BASE_DIR="../cryofold/cryobench_IgD/IgD_remi/"
RUN_DIR=$BASE_DIR/run

cryodrgn parse_ctf_star $BASE_DIR/particles_*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/Particles/particles.mrcs --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject

python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose_auto --backproject_path $BASE_DIR/backproject/backproject.mrc\
  --embedding_pdb_path data/cryofold/cryobench_IgD/IgG-1D/images/snr0.01/run_target/fit.4000.pdb \
    --dist_search 5 --grid_size 100 --pixel_size 3.0 --sigma 1.05 --real_space --overwrite