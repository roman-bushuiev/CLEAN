#!/bin/bash

python3 CLEAN_inference.py \
    --inference_fasta /home/bushuant/colabfold/enzyme-datasets/all_sequences.fasta \
    --out_embs_dir data/embeddings/enzyme-datasets/

# python -u CLEAN_inference.py \
#     --inference_fasta_folder data \
#     --inference_fasta new.fasta \
#     --gpu_id 0 \
#     --inference_fasta_start 0 \
#     --inference_fasta_end 250 \
#     --toks_per_batch 2048 \
#     --esm_batches_per_clean_inference 300 \