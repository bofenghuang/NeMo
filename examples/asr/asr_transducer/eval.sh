#!/usr/bin/env bash

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=2

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"


python ${NEMO_GIT_FOLDER}/examples/asr/asr_transducer/eval.py \
    --config-path="../conf/conformer" --config-name="conformer_transducer_bpe" \
    name="stt_fr_conformer_transducer_large" \
    model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json" \
    model.train_ds.max_duration=30 \
    model.train_ds.min_duration=1 \
    model.validation_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm.json" \
    model.test_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm.json" \
    model.spec_augment.time_masks=2 \
    model.tokenizer.dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/tokenizers/tokenizer_spe_bpe_v128" \
    model.tokenizer.type="bpe" \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    model.train_ds.batch_size=8 \
    model.validation_ds.batch_size=8 \
    model.decoding.strategy="beam" \
    model.decoding.beam.beam_size=10 \
    trainer.accumulate_grad_batches=32 \
    trainer.precision=16 \
    trainer.max_epochs=100 \
    trainer.val_check_interval=0.5 \
    exp_manager.resume_if_exists=True \
    +exp_manager.explicit_log_dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft" \
