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

