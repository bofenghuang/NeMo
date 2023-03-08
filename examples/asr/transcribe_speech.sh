#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=2

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# modelpath="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_new/checkpoints/stt_fr_conformer_transducer_large.nemo"
# modelpath="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/checkpoints/stt_fr_conformer_transducer_large.nemo"
modelpath="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/checkpoints/stt_fr_conformer_transducer_large-averaged.nemo"

# testmanifest="/home/bhuang/asr/NeMo/examples/asr/tmp/test_hmhm.json"
# testmanifest="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm.json"
# testmanifest="/projects/corpus/voice/zaion/lbpa/2023-02-21/data/nemo_manifest/data_without_partial_words.json"
testmanifest="/projects/corpus/voice/zaion/lbpa/2023-02-22/data/nemo_manifest/data_without_partial_words.json"

# audio_dir="" \

# outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/outputs_averaged/test_hmhm_greedy"
# outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/outputs_averaged/test_hmhm_beam10"
# outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/outputs_averaged/test_lbpa_greedy"
outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/outputs_averaged/test_lbpa_lot2_beam10"

decode_opts=(rnnt_decoding.strategy="beam" rnnt_decoding.beam.beam_size=10)
# rnnt_decoding.strategy="beam" \
# rnnt_decoding.beam.beam_size=10 \

python ${NEMO_GIT_FOLDER}/examples/asr/transcribe_speech.py \
    model_path=$modelpath \
    dataset_manifest=$testmanifest \
    output_filename="$outdir/predictions.json" \
    batch_size=16 \
    num_workers=4 \
    compute_langs=False \
    cuda=0 \
    amp=True \
    ${decode_opts[@]} \
    append_pred=False \
    pred_name_postfix=""

# eval predictions
python ${NEMO_GIT_FOLDER}/examples/asr/eval.py --dataset_manifest_path "$outdir/predictions.json" --out_dir $outdir --do_ignore_words False
python ${NEMO_GIT_FOLDER}/examples/asr/eval.py --dataset_manifest_path "$outdir/predictions.json" --out_dir $outdir --do_ignore_words True
