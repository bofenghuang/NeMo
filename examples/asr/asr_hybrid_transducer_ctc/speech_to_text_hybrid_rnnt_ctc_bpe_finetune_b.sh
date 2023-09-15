#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

"""
Finetune stt_fr_conformer_transducer_large on
"""

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=1

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
# model.tokenizer.type=<either bpe or wpe> \

# ~trainer.strategy \
# trainer.strategy="ddp" \

# +init_from_nemo_model.model.path
# +init_from_nemo_model.model.path="/home/bhuang/.cache/torch/NeMo/NeMo_1.13.0rc0/stt_fr_conformer_transducer_large/0afcc58c13c5341db452f7a37e5ee0bd/stt_fr_conformer_transducer_large.nemo" \
# +init_from_pretrained_model="stt_fr_conformer_transducer_large" \

python ${NEMO_GIT_FOLDER}/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="../conf/conformer/hybrid_transducer_ctc" --config-name="conformer_hybrid_transducer_ctc_bpe" \
    name="stt_fr_asr_hybrid_transducer_large" \
    +init_from_nemo_model.model.path="/home/bhuang/.cache/torch/NeMo/NeMo_1.13.0rc0/stt_fr_conformer_transducer_large/0afcc58c13c5341db452f7a37e5ee0bd/stt_fr_conformer_transducer_large.nemo" \
    model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw_wo_space_after_apostrophe.json" \
    model.train_ds.batch_size=8 \
    model.train_ds.use_start_end_token=True \
    model.train_ds.trim_silence=True \
    model.train_ds.max_duration=30 \
    model.validation_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm_wo_space_after_apostrophe.json" \
    model.validation_ds.batch_size=8 \
    model.validation_ds.use_start_end_token=True \
    +model.validation_ds.trim_silence=True \
    model.test_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/test_hmhm_wo_space_after_apostrophe.json" \
    model.test_ds.batch_size=8 \
    model.test_ds.use_start_end_token=True \
    +model.test_ds.trim_silence=True \
    model.tokenizer.dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/tokenizers/tokenizer_spe_bpe_v128" \
    model.tokenizer.type="bpe" \
    model.aux_ctc.ctc_loss_weight=0.3 \
    model.joint.fused_batch_size=8 \
    model.optim.name="adamw" \
    model.optim.lr=0.0001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.name="CosineAnnealing" \
    ~model.optim.sched.d_model \
    model.optim.sched.warmup_steps=2000 \
    model.spec_augment.time_masks=2 \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    ~trainer.strategy \
    trainer.accumulate_grad_batches=32 \
    trainer.precision=16 \
    trainer.max_epochs=32 \
    trainer.val_check_interval=0.5 \
    +exp_manager.explicit_log_dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_asr_hybrid_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe" \
    exp_manager.resume_if_exists=True \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="asr-hybrid-transducer-hmhm-merged-and-raw-ft_pretrained_bpe" \
    exp_manager.wandb_logger_kwargs.project="nemo-asr-hmhm"
