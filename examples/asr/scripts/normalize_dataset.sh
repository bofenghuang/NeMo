#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Normalize text in NeMo's manifest files.
# Run analysis on each sub dataset.

export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"
export DATASETS_VERBOSITY="error"

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# Normalize text (lowercase, num2text, remove punctuation)
####################################################################################################
# inputfiles="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/train/train_mozilla-foundation_common_voice_13_0_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/validation/validation_mozilla-foundation_common_voice_13_0_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/test/test_mozilla-foundation_common_voice_13_0_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/validation/validation_facebook_multilingual_librispeech_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/test/test_facebook_multilingual_librispeech_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/validation/validation_facebook_voxpopuli_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/train/train_google_fleurs_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/test/test_google_fleurs_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/test/test_gigant_african_accented_french_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/PolyAI/minds14/fr-FR/train/train_PolyAI_minds14_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/media_speech/FR/media_speech_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/m_ailabs/FR/m_ailabs_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/train/train_mtedx_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/valid/valid_mtedx_manifest.json
# /projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/test/test_mtedx_manifest.json"

# # inputfiles="/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/train/train_mtedx_manifest.json
# # /projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/valid/valid_mtedx_manifest.json
# # /projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/test/test_mtedx_manifest.json"

# for inputfile in $inputfiles; do
#     # echo $inputfile
#     python ${NEMO_GIT_FOLDER}/examples/asr/scripts/normalize_dataset.py $inputfile
# done

# for inputfile in $inputfiles; do
#     # echo $inputfile
#     inputfile=${inputfile%.*}
#     inputfile=${inputfile}_normalized.json
#     # echo $inputfile

#     python ${NEMO_GIT_FOLDER}/examples/asr/scripts/analyze_dataset.py $inputfile
# done
####################################################################################################

# Normalize text (PnC, text2num)
####################################################################################################

inputfiles="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/train/train_mozilla-foundation_common_voice_13_0_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/validation/validation_mozilla-foundation_common_voice_13_0_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/test/test_mozilla-foundation_common_voice_13_0_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/validation/validation_facebook_voxpopuli_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/train/train_google_fleurs_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/test/test_google_fleurs_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/m_ailabs/FR/m_ailabs_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/train/train_mtedx_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/valid/valid_mtedx_manifest.json
/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/test/test_mtedx_manifest.json"

for inputfile in $inputfiles; do
    # echo $inputfile
    python ${NEMO_GIT_FOLDER}/examples/asr/scripts/normalize_dataset.py $inputfile "text" "_normalized_pnc"
done

# python scripts/merge_datasets.py
# python scripts/preprocess_dataset.py /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/train_asr.json /projects/bhuang/corpus/speech/nemo_manifests/final/2023-11-07/train_asr_processed_dedup256.json --max_identical_text 256


####################################################################################################
