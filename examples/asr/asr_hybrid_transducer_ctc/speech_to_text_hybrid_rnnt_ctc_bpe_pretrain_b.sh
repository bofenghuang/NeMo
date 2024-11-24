#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Train stt_fr_fastconformer_hybrid_transducer_ctc_bpe model from scratch on public data

export HYDRA_FULL_ERROR=1

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"

export CUDA_VISIBLE_DEVICES="3,4"

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# tokenizer
tokenizer_dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/tokenizers_general/tokenizer_spe_unigram_v1024"
# train data manifest
# train_manifest_filepath="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256.json"
# train tarred data
tardir="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256_tarred"
train_manifest_filepath="[[$tardir/bucket1/tarred_audio_manifest.json],[$tardir/bucket2/tarred_audio_manifest.json],[$tardir/bucket3/tarred_audio_manifest.json],[$tardir/bucket4/tarred_audio_manifest.json]]"
train_tarred_audio_filepaths="[[$tardir/bucket1/audio__OP_0..7_CL_.tar],[$tardir/bucket2/audio__OP_0..7_CL_.tar],[$tardir/bucket3/audio__OP_0..7_CL_.tar],[$tardir/bucket4/audio__OP_0..7_CL_.tar]]"
# valida data manifedt
validation_manifest_filepath="[/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/validation/validation_mozilla-foundation_common_voice_13_0_manifest_normalized.json,/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/validation/validation_facebook_multilingual_librispeech_manifest_normalized.json,/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/validation/validation_facebook_voxpopuli_manifest_normalized.json,/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest_normalized.json,/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/valid/valid_mtedx_manifest_normalized.json]"
# output
outdir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_fastconformer_hybrid_transducer_ctc_bpe/large_bs2048_lr1e3"

wandb_name="${outdir##*/}"

# tips:
# If fp16 is not stable and model diverges after some epochs, you may use fp32.
# Default learning parameters in this config are set for global batch size of 2K while you may use lower values.
# hydra override: https://hydra.cc/docs/advanced/override_grammar/basic

    # model.train_ds.bucketing_batch_size=64 \
    # model.train_ds.bucketing_strategy="fully_randomized" \
    # exp_manager.resume_if_exists=True \

python ${NEMO_GIT_FOLDER}/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="../conf/fastconformer/hybrid_transducer_ctc" --config-name="fastconformer_hybrid_transducer_ctc_bpe" \
    name="stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large" \
    model.train_ds.manifest_filepath=$train_manifest_filepath \
    model.train_ds.is_tarred=True \
    model.train_ds.tarred_audio_filepaths=$train_tarred_audio_filepaths \
    model.train_ds.shuffle_n=528387 \
    model.train_ds.batch_size=32 \
    model.train_ds.max_duration=30 \
    model.train_ds.min_duration=0.1 \
    model.validation_ds.manifest_filepath=$validation_manifest_filepath \
    model.validation_ds.batch_size=32 \
    +model.validation_ds.max_duration=30 \
    +model.validation_ds.min_duration=0.1 \
    ~model.test_ds \
    model.tokenizer.dir=$tokenizer_dir \
    model.tokenizer.type="bpe" \
    model.optim.lr=0.001 \
    model.optim.sched.name="CosineAnnealing" \
    ~model.optim.sched.d_model \
    model.optim.sched.warmup_steps=5000 \
    model.optim.sched.min_lr=0.0001 \
    trainer.max_epochs=100 \
    trainer.accelerator="gpu" \
    trainer.accumulate_grad_batches=32 \
    trainer.precision=16 \
    exp_manager.exp_dir=$outdir \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=$wandb_name \
    exp_manager.wandb_logger_kwargs.project="nemo-asr-general"
