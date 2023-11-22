#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

"""
Merge NeMo's manifest files into one.
"""

import os

os.environ["HF_DATASETS_CACHE"] = "/projects/bhuang/.cache/huggingface/datasets"
os.environ["DATASETS_VERBOSITY"] = "error"

import sys

sys.path.append("/home/bhuang/myscripts")
from datasets import concatenate_datasets, load_dataset
from hf_dataset_processing.file_utils import write_dataset_to_json

# input_file_paths = [
#     "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/train/train_mozilla-foundation_common_voice_13_0_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/train/train_google_fleurs_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/PolyAI/minds14/fr-FR/train/train_PolyAI_minds14_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/media_speech/FR/media_speech_manifest_normalized.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/m_ailabs/FR/m_ailabs_manifest_normalized.json",
#     # "/projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized.json",
#     # "/projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized_dedup256.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized_min1_dedup256.json",
#     # "/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized.json",
#     # "/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_dedup4.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_min05_dedup4.json",
#     "/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/train/train_mtedx_manifest_normalized.json",
# ]
# output_file = "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr.json"

input_file_paths = [
    "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/train/train_mozilla-foundation_common_voice_13_0_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/validation/validation_mozilla-foundation_common_voice_13_0_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/validation/validation_facebook_voxpopuli_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/train/train_google_fleurs_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/m_ailabs/FR/m_ailabs_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/train/train_mtedx_manifest_normalized_pnc.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/valid/valid_mtedx_manifest_normalized_pnc.json",
    # weak PnC
    "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest_normalized_pnc_cleaned_filtered.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/validation/validation_facebook_multilingual_librispeech_manifest_normalized_pnc_cleaned_filtered.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest_normalized_pnc_cleaned.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/media_speech/FR/media_speech_manifest_normalized_pnc_cleaned.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/PolyAI/minds14/fr-FR/train/train_PolyAI_minds14_manifest_normalized_pnc_cleaned.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized_min1_dedup256_pnc_cleaned.json",
    # "/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_min05_dedup4_pnc_cleaned.json",
    "/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_min5_mindur05_dedup4_pnc_cleaned.json",
]
output_file = "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/train_asr.json"

input_file_paths = [
    "/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/test/test_mozilla-foundation_common_voice_13_0_manifest_normalized_pnc.json",
]
output_file = "/projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-21/test_asr_mcv13_manifest_normalized_pnc.json"

column_names = {"text", "audio_filepath", "duration"}

datasets = []
for input_file_path in input_file_paths:
    ds = load_dataset("json", data_files=input_file_path)["train"]
    ds = ds.remove_columns([col for col in ds.column_names if col not in column_names])
    # add subset mark
    ds = ds.map(lambda _: {"split": input_file_path.rsplit("/", 1)[-1].split(".", 1)[0]}, num_proc=16)

    datasets.append(ds)

final_ds = concatenate_datasets(datasets)
print(final_ds)

print(f'Total duration: {sum(final_ds["duration"]) / 3600:.2f}h')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

write_dataset_to_json(final_ds, output_file)
print(f"The processed data is saved into {output_file}")
