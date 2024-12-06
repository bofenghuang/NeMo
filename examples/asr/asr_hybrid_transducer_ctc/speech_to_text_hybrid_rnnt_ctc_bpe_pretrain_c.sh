#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Train stt_fr_fastconformer_hybrid_transducer_ctc_bpe model from scratch on public data

# install debs
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install -e .[asr]
# conda remove numba
# pip uninstall numba
# conda install -c conda-forge numba
# pip install numpy==1.26.4

export HYDRA_FULL_ERROR=1

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="2,3"

NEMO_GIT_FOLDER="/home/bhuang/NeMo"

data_root="/projects/bhuang/corpus/speech/nemo_manifests"

train_files=(
    # fr
    # "${data_root}/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped.json"
    # "${data_root}/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # "${data_root}/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # es
    "${data_root}/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped.json"
    # _lang -> lang
    # "${data_root}/tmp/train_mls_fr.json"
    # "${data_root}/tmp/train_mls_es.json"

)

validation_files=(
    "${data_root}/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest.json"
    # "${data_root}/tmp/valid_fleurs_fr.json"
    # "${data_root}/tmp/valid_fleurs_es.json"
)

train_manifest_filepath=$(IFS=,; echo "${train_files[*]}")
validation_manifest_filepath=$(IFS=,; echo "${validation_files[*]}")

# tokenize cumstomized dataset
# python ${NEMO_GIT_FOLDER}/scripts/tokenizers/process_asr_text_tokenizer.py \
#     --manifest="${train_manifest_filepath}" \
#     --data_root="${NEMO_GIT_FOLDER}/examples/asr/nemo_experiments/tokenizers/es" \
#     --vocab_size=1024 \
#     --tokenizer="spe" \
#     --spe_type="bpe" \
#     --spe_character_coverage=0.9995 \
#     --spe_remove_extra_whitespaces \
#     --no_lower_case \
#     --log

# exit;

# tokenizer
# tokenizer_dir="${NEMO_GIT_FOLDER}/examples/asr/nemo_experiments/tokenizers/tokenizer_spe_bpe_v1024"

# train data manifest
# train_manifest_filepath="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256.json"
# valida data manifest
# validation_manifest_filepath="/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest.json"

# output
outdir="${NEMO_GIT_FOLDER}/examples/asr/nemo_experiments/tmp_agg"

wandb_name="${outdir##*/}"

    # model.tokenizer.dir="$tokenizer_dir" \
    # model.tokenizer.type="bpe" \

    # +model.train_ds.augmentor.white_noise.prob=1.0 \
    # +model.train_ds.augmentor.white_noise.min_level=-90 \
    # +model.train_ds.augmentor.white_noise.max_level=-46 \
    # +model.train_ds.augmentor.shift.prob=1.0 \
    # +model.train_ds.augmentor.shift.min_shift_ms=-5.0 \
    # +model.train_ds.augmentor.shift.max_shift_ms=5.0 \
    # model.optim.weight_decay=0.0001 \

python ${NEMO_GIT_FOLDER}/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="../conf/fastconformer/hybrid_transducer_ctc" --config-name="fastconformer_hybrid_transducer_ctc_bpe" \
    name="stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large" \
    model.train_ds.manifest_filepath="'$train_manifest_filepath'" \
    model.train_ds.batch_size=32 \
    model.train_ds.num_workers=16 \
    model.train_ds.max_duration=30 \
    model.train_ds.min_duration=0.1 \
    model.validation_ds.manifest_filepath="'$validation_manifest_filepath'" \
    model.validation_ds.batch_size=32 \
    model.validation_ds.num_workers=16 \
    +model.validation_ds.max_duration=30 \
    +model.validation_ds.min_duration=0.1 \
    ~model.test_ds \
    model.tokenizer.type="agg" \
    +model.tokenizer.langs.fr.type="bpe" \
    +model.tokenizer.langs.fr.dir="/home/bhuang/NeMo/examples/asr/nemo_experiments/tokenizers/fr/tokenizer_spe_bpe_v1024" \
    +model.tokenizer.langs.es.type="bpe" \
    +model.tokenizer.langs.es.dir="/home/bhuang/NeMo/examples/asr/nemo_experiments/tokenizers/es/tokenizer_spe_bpe_v1024" \
    model.spec_augment.freq_masks=0 \
    model.spec_augment.time_masks=0 \
    model.optim.lr=0.001 \
    model.optim.sched.name="CosineAnnealing" \
    ~model.optim.sched.d_model \
    ~model.optim.sched.warmup_steps \
    model.optim.sched.warmup_ratio=0.05 \
    model.optim.sched.min_lr=0.0001 \
    trainer.max_epochs=100 \
    trainer.val_check_interval=0.5 \
    trainer.accelerator="gpu" \
    trainer.num_nodes=1 \
    trainer.accumulate_grad_batches=4 \
    trainer.precision="bf16-mixed" \
    exp_manager.exp_dir="$outdir" \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="$wandb_name" \
    exp_manager.wandb_logger_kwargs.project="nemo-asr-general"
