#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=logs/%x/%j.out                 # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err                  # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16                      # number of cores per tasks
#SBATCH --time=20:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3                        # QoS
#SBATCH --hint=nomultithread                    # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --account=cjc@v100

set -x -e

module purge
module load git-lfs
module load unrar
module load anaconda-py3/2023.03
module load cuda/12.1.0
# conda activate speech
conda activate asr

# Convert datasets on Hugging Face Hub or local datasets to NeMo's manifest format.
# Also resample and export to wav files.

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# export HF_HOME="/projects/bhuang/.cache/huggingface"
export REUSE_DATASET_IF_EXISTS="false"

# NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

# outdir="/projects/bhuang/corpus/speech/nemo_manifests"
# outdir="$ALL_CCFRWORK/corpus/speech/nemo_manifests"
outdir="$ALL_CCFRSCRATCH/corpus/speech/nemo_manifests"

num_proc="64"

# mcv
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="de" \
#     split="train" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="de" \
#     split="validation" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# mls
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="german" \
#     split="train" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# voxpopuli
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="de" \
#     split="train" \
#     text_column_name="raw_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# fleurs
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="google/fleurs" \
#     name="de_de" \
#     split="validation" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# yodas
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="de000" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="de100" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="espnet/yodas" \
    name="de101" \
    split="train" \
    text_column_name="text" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="de102" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"
