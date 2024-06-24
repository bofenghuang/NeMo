#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

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

# export HF_HOME="/projects/bhuang/.cache/huggingface"

# NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

# outdir="/projects/bhuang/corpus/speech/nemo_manifests"
# outdir="$ALL_CCFRWORK/corpus/speech/nemo_manifests"
outdir="$ALL_CCFRSCRATCH/corpus/speech/nemo_manifests"

num_proc="32"

# mcv
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="it" \
#     split="train" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="it" \
#     split="validation" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="it" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# exit 0;

# mls
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="italian" \
#     split="train" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="italian" \
#     split="validation" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="italian" \
#     split="test" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# voxpopuli
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="it" \
#     split="train" \
#     text_column_name="raw_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="it" \
#     split="validation" \
#     text_column_name="raw_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="it" \
#     split="test" \
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
#     name="it_it" \
#     split="train" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="google/fleurs" \
#     name="it_it" \
#     split="validation" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="google/fleurs" \
#     name="it_it" \
#     split="test" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# exit 0;

# media-speech
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/media-speech/data.tsv" \
#     path="media_speech" \
#     name="FR" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# mtedx
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/train_asr.tsv" \
#     path="mtedx" \
#     name="fr-fr" \
#     split="train" \
#     text_column_name="tgt_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/valid_asr.tsv" \
#     path="mtedx" \
#     name="fr-fr" \
#     split="valid" \
#     text_column_name="tgt_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/multilingual-tedx/fr-fr/test_asr.tsv" \
#     path="mtedx" \
#     name="fr-fr" \
#     split="test" \
#     text_column_name="tgt_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# african_accented_french
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="gigant/african_accented_french" \
#     split="train" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="gigant/african_accented_french" \
#     split="test" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# PolyAI/minds14
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="PolyAI/minds14" \
#     name="fr-FR" \
#     split="train" \
#     text_column_name="transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# m-ailabs
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/m-ailabs/data.tsv" \
#     path="m_ailabs" \
#     name="FR" \
#     text_column_name="clean_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# att-hack
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/att-hack/data.tsv" \
#     path="att_hack" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# lingualibre
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     local_file="/projects/bhuang/corpus/speech/lingualibre/Q21-fra-French/data.tsv" \
#     path="lingualibre" \
#     name="FR" \
#     num_proc="64" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# hf eval data
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="speech-recognition-community-v2/dev_data" \
#     name="fr" \
#     split="validation" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     use_auth_token="True"

# yodas
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="it000" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="it100" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas" \
#     name="it101" \
#     split="train" \
#     text_column_name="text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"
