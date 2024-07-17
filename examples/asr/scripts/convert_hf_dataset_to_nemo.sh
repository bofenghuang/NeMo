#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Convert datasets on Hugging Face Hub or local datasets to NeMo's manifest format.
# Also resample and export to wav files.

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
export OMP_NUM_THREADS="1"

export HF_HOME="/projects/bhuang/.cache/huggingface"

NEMO_GIT_FOLDER="/home/bhuang/NeMo"

outdir="/projects/bhuang/corpus/speech/nemo_manifests"

num_proc="16"

# mcv
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="train" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="validation" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

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

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="other" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="invalidated" \
#     text_column_name="sentence" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="mozilla-foundation/common_voice_17_0" \
#     name="fr" \
#     split="validated" \
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
#     name="french" \
#     split="train" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="french" \
#     split="validation" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/multilingual_librispeech" \
#     name="french" \
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
#     name="fr" \
#     split="train" \
#     text_column_name="raw_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="fr" \
#     split="validation" \
#     text_column_name="raw_text" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="facebook/voxpopuli" \
#     name="fr" \
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
#     name="fr_fr" \
#     split="train" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="google/fleurs" \
#     name="fr_fr" \
#     split="validation" \
#     text_column_name="raw_transcription" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"

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

# yodas2
# python ${NEMO_GIT_FOLDER}/scripts/speech_recognition/convert_hf_dataset_to_nemo_b.py \
#     output_dir="$outdir" \
#     path="espnet/yodas2" \
#     name="fr000" \
#     split="train" \
#     text_column_name="utterances" \
#     num_proc="$num_proc" \
#     ensure_ascii="False" \
#     trust_remote_code="True" \
#     use_auth_token="True"
