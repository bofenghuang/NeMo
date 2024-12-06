#!/bin/bash
# Copyright 2024  Bofeng Huang

#SBATCH --job-name=train-tok
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-dev             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# Train tokenizer

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
# module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda activate speech
# conda activate asr
conda activate nemo

# NEMO_GIT_FOLDER="/home/bhuang/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

lang=$1

# input_root=/projects/bhuang/corpus/speech/nemo_manifests
input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

train_files=(
    # $input_root/final_nemo/train_manifest_it_wer20.json
    # $input_root/final_nemo/train_manifest_${lang}_wer20.json
    # $input_root/final_nemo/train_manifest_${lang}_wer10.json
    # whisper
    $input_root/final_nemo_whisper/train_manifest_${lang}_wer20.json
    # $input_root/final_nemo_whisper/train_manifest_${lang}_wer10.json
)

train_manifest_filepath=$(IFS=,; echo "${train_files[*]}")

# output_root="/lustre/fswork/projects/rech/gkb/commun/outputs/nemo_experiments/tokenizers"
# whisper
output_root="/lustre/fswork/projects/rech/gkb/commun/outputs/nemo_experiments/tokenizers_whisper"

# tokenize cumstomized dataset
python ${NEMO_GIT_FOLDER}/scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="${train_manifest_filepath}" \
    --data_root="${output_root}/${lang}" \
    --vocab_size=1024 \
    --tokenizer="spe" \
    --spe_type="bpe" \
    --spe_character_coverage=0.9995 \
    --spe_remove_extra_whitespaces \
    --no_lower_case \
    --log

echo "END TIME: $(date)"