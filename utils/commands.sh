
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







BASE_DIR="../cryofold/cryobench_IgD/IgG-1D/images/snr0.01"
RUN_DIR=$BASE_DIR/run2

cryodrgn parse_ctf_star $BASE_DIR/particles_*.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_*.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/sorted_particles.128.txt  --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject


python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose \
 --from_aligned_pdb ~/cryofold/cryobench_IgD/pred2/aligned.pdb --alignment_reference ~/cryofold/cryobench_IgD/pred2/unrelaxed.pdb  --overwrite

nohup \
python ./scripts/train.py $BASE_DIR/000_particles_128.mrcs  \
    --poses  $BASE_DIR/particles_000.pkl\
    --ctf $BASE_DIR/ctf_000.pkl \
    -n 50 \
    -o $RUN_DIR \
    --pixel_size 3.0 \
    --sigma 1.05 \
    --quality_ratio 4.0 \
    --af_decoder \
    --embedding_path  ~/cryofold/cryobench_IgD/pred2/embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose_000.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 1  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 256 \
    --enc-layers 3 \
    --dec-dim 128 \
    --dec-layers 3 \
    --domain hartley \
    --overwrite \
    --multimer \
    --center_loss_weight 0.0001\
    --debug \
    > log.txt &
    # --all_atom \



cryodrgn parse_ctf_star $BASE_DIR/snr0.01_000.star -o $BASE_DIR/ctf_000.pkl
cryodrgn parse_pose_star $BASE_DIR/snr0.01_000.star -o $BASE_DIR/particles_000.pkl
cryodrgn backproject_voxel $BASE_DIR/000_particles_128.mrcs  --poses $BASE_DIR/particles_000.pkl --ctf $BASE_DIR/ctf_000.pkl -o $BASE_DIR/backproject_000

python ./scripts/compute_initial_pose.py $BASE_DIR/initial_pose_000 \
 --from_aligned_pdb ~/cryofold/cryobench_IgD/pred2/aligned.pdb --alignment_reference ~/cryofold/cryobench_IgD/pred2/unrelaxed.pdb  --overwrite


python ./scripts/analyze.py -o $RUN_DIR/analysis $RUN_DIR 19 --pc 2



python ./scripts/train.py $BASE_DIR/000_particles_128.mrcs     \
 --poses  $BASE_DIR/particles_000.pkl    --ctf $BASE_DIR/ctf_000.pkl     \
 -n 50     -o $RUN_DIR     --pixel_size 3.0     --sigma 1.05   \
   --quality_ratio 4.0     --af_decoder     --embedding_path  ~/cryofold/cryobench_IgD/pred2/embeddings.pt     \
   --initial_pose_path $BASE_DIR/initial_pose_000.pt     --af_checkpoint_path  ../openfold/openfold/resources/params/params_model_3_multimer_v3.npz     \
   --batch-size 1      --num-workers 0     --zdim 4      --enc-dim 256     --enc-layers 3     --dec-dim 128     --dec-layers 3     --domain hartley     \
   --overwrite     --multimer     --center_loss_weight 0.0001  --all_atom --debug --device 1
