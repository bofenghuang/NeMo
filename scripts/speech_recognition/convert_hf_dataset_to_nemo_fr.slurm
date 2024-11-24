#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=logs/%x/%j.out                 # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err                  # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                     # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4                       # number of cores per tasks
#SBATCH --time=20:00:00                         # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3                        # QoS
#SBATCH --hint=nomultithread                    # we get physical cores not logical
#SBATCH --partition=prepost
#SBATCH --account=cjc@cpu

set -x -e

# set up environment
module purge
# module load arch/h100
module load git-lfs
module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda activate speech
conda activate asr

# Convert datasets on Hugging Face Hub or local datasets to NeMo's manifest format.
# Also resample and export to wav files.

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export REUSE_DATASET_IF_EXISTS="false"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
# export HF_HUB_ETAG_TIMEOUT="60"
# export HF_HUB_DOWNLOAD_TIMEOUT="60"
# https://github.com/huggingface/transformers/issues/17611
# export CURL_CA_BUNDLE=""

# NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

# outdir="/projects/bhuang/corpus/speech/nemo_manifests"
# outdir="$ALL_CCFRWORK/corpus/speech/nemo_manifests"
# outdir="$ALL_CCFRSCRATCH/corpus/speech/nemo_manifests"
outdir="$ALL_CCFRSCRATCH/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual"

num_proc="64"

# mcv
python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="mozilla-foundation/common_voice_17_0" \
    name="fr" \
    split="train" \
    text_column_name="sentence" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="mozilla-foundation/common_voice_17_0" \
    name="fr" \
    split="validation" \
    text_column_name="sentence" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# mls
python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="facebook/multilingual_librispeech" \
    name="french" \
    split="train" \
    text_column_name="transcript" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="french" \
#     split="test" \
#     text_column_name="transcript" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# voxpopuli
python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="facebook/voxpopuli" \
    name="fr" \
    split="train" \
    text_column_name="normalized_text" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="fr" \
#     split="test" \
#     text_column_name="normalized_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# fleurs
python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
    output_dir="$outdir" \
    path="google/fleurs" \
    name="fr_fr" \
    split="validation" \
    text_column_name="raw_transcription" \
    num_proc="$num_proc" \
    ensure_ascii="False" \
    trust_remote_code="True" \
    use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="google/fleurs" \
#     name="fr_fr" \
#     split="test" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# african_accented_french
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="gigant/african_accented_french" \
#     split="train" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="gigant/african_accented_french" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# yodas
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="fr000" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="fr100" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="fr101" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="fr102" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="fr103" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# hf eval data
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="speech-recognition-community-v2/dev_data" \
#     name="fr" \
#     split="validation" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# hf
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="eustlb/french-long-form-test" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# Accueil_UBS
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="BrunoHays/Accueil_UBS" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"
