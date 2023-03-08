#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=2

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# modelpath="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft/checkpoints/stt_fr_conformer_transducer_large.nemo"
modelpath="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_ctc_large/hmhm_ft/checkpoints/conformer-ctc-bpe-fr-finetune-hmhm.nemo"
testmanifest="/home/bhuang/asr/NeMo/examples/asr/tmp/test_hmhm.json"
# testmanifest="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm.json"
outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft/outputs/test_hmhm_ngram_beam10"
probscachefile="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft/outputs/probs_cache_file"


python ${NEMO_GIT_FOLDER}/examples/asr/eval_beamsearch_ngram.py \
    --nemo_model_file $modelpath \
    --kenlm_model_file <path to the binary KenLM model> \
    --input_manifest $testmanifest \
    --preds_output_folder $outdir \
    --probs_cache_file $probscachefile \
    --acoustic_batch_size 16 \
    --device 0 \
    --use_amp \
    --decoding_mode "beamsearch_ngram" \
    --beam_width 10 \
    --beam_alpha 0.5 \
    --beam_beta 1.0
