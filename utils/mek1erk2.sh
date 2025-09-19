BASE_DIR="/home/vuillemr/flexfold/data/cryofold/jillsData/particles"
RUN_DIR=$BASE_DIR/run


cryodrgn parse_ctf_star $BASE_DIR/particles_ctf.star -o $BASE_DIR/ctf.pkl
cryodrgn parse_pose_star $BASE_DIR/particles_ctf.star -o $BASE_DIR/particles.pkl
cryodrgn backproject_voxel $BASE_DIR/particles.txt --poses $BASE_DIR/particles.pkl --ctf $BASE_DIR/ctf.pkl -o $BASE_DIR/backproject



python ./scripts/train.py $BASE_DIR/Particles/particles.mrcs  \
    --poses $BASE_DIR/particles.pkl \
    --ctf $BASE_DIR/ctf.pkl \
    -n 100 \
    -o $RUN_DIR \
    --pixel_size 2.2 \
    --sigma 1.05 \
    --quality_ratio 5.0 \
    --embedding_path ../cryofold/embeddings/4ake_A_embeddings.pt \
    --initial_pose_path $BASE_DIR/initial_pose.pt \
    --af_checkpoint_path  ../openfold/openfold/resources/openfold_params/finetuning_no_templ_1.pt \
    --batch-size 64  \
    --num-workers 0 \
    --zdim 4  \
    --enc-dim 64 \
    --enc-layers 4 \
    --dec-dim 64 \
    --dec-layers 4 \
    --domain real \
    --encode-mode conv \
    --all_atom\
    --overwrite\
    --frozen_structure_module\
    --pair_stack
python ./scripts/analyze.py  -o $RUN_DIR/analysis $RUN_DIR 99 --pc 2 --trajectory
python ./scripts/compare_traj.py "$RUN_DIR/analysis/traj/*pdb" "$BASE_DIR/gt_pdbs/*pdb" $RUN_DIR/analysis/stats.txt