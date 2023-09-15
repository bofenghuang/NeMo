#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang

export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=1

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

# prep nemo manifest and upsample & save audio segments
# ./prep_data_with_upsampling.py \
#     /home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv \
#     /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
#     /home/bhuang/corpus/speech/internal/hm_hm_16k/audios_16k/train_hmhm_merged_and_raw

# tokenize cumstomized dataset
# bh: vocabs by --spe_type="unigram" are weird
# python ${NEMO_GIT_FOLDER}/scripts/tokenizers/process_asr_text_tokenizer.py \
#     --manifest="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm.json" \
#     --data_root="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/tokenizers" \
#     --vocab_size=128 \
#     --tokenizer="spe" \
#     --spe_type="bpe" \
#     --spe_character_coverage=1.0 \
#     --no_lower_case \
#     --log
# exit

# bh: Freeze encoder
# Freezing the encoder is generally helpful to limit computation and enable faster training
# however, in many experiments, freezing the encoder in its entirety will often prevent a model from learning on low-resource languages.
# In order to enable a frozen encoder model to learn on a new language stably, we, therefore, unfreeze the batch normalization layers in the encoder.
# On top of this, if the model contains "SqueezeExcite" sub-modules, we unfreeze them as well.
# In doing so, we notice that such models train properly and obtain respectable scores even on severely resource-limited languages.
# This phenomenon disappears when sufficient data is available (in such a case, the entire encoder can be trained as well).
# Therefore it is advised to unfreeze the encoder when sufficient data is available.

# bh: reload decoder and joint
# if we choose the same vocab size with the pretrained one, don't need to exclude these, and get the fastest convergence
# https://github.com/NVIDIA/NeMo/discussions/3725
# +init_from_nemo_model.model.exclude: ["decoder.prediction.embed.weight","joint.joint_net.2.weight","joint.joint_net.2.bias"]

# bh: Char vs BPE
# In general, there are minor differences between the Character encoding and Sub-word encoding models.
# Since sub-words can encode larger sequence of tokens into a single subword, they substantially reduce the target sequence length.
# Citrinet takes advantage of this reduction by aggressively downsampling the input three times (a total of 8x downsampling).
# At this level of downsampling, it is possible to encounter a specific limitation of CTC loss.
# CTC loss works under the assumption that T (the acoustic model's output sequence length) > U (the target sequence length).
# If this criterion is violated, CTC loss is practically set to inf (which is then forced to  0  by PyTorch's zero_infinity flag), and its gradient is set to 0.
# Therefore it is essential to inspect the ratio of T / U and ensure that it's reasonably close to 1 or higher.

# bh: lr NoamAnnealing is used for training from scratch, use something else here for finetuning

# bh: decoding strategies
# 1) greedy: This is sample-level greedy decoding. It is generally exceptionally slow as each sample in the batch will be decoded independently.
# For publications, this should be used alongside batch size of 1 for exact results.
# - max_symbols: maximum number of target token decoding per acoustic timestep.
# Note that during training, this was implicitly constrained by the shape of the joint matrix (max_symbols = U). 
# However, there is no such U upper bound during inference (we don't have the ground truth U).
# So we explicitly set a heuristic upper bound on how many decoding steps can be performed per acoustic timestep.
# Generally a value of 5 and above is sufficient.
# 2) greedy_batch: This is the general default and should nearly match the greedy decoding scores (if the acoustic features are not affected by feature mixing in batch mode).
# Even for small batch sizes, this strategy is significantly faster than greedy.
# 3) beam: Runs beam search with the implicit language model of the Prediction model.
# It will generally be quite slow, and might need some tuning of the beam size to get better transcriptions.
# - beam_size: Since this is implemented in PyTorch, large beam sizes will take exorbitant amounts of time.
# - score_norm: Whether to normalize scores prior to pruning the beam.
# - return_best_hypothesis: If beam search is being performed, we can choose to return just the best hypothesis or all the hypotheses.
# - tsd_max_sym_exp: The maximum symmetric expansions allowed per timestep during beam search. Larger values should be used to attempt decoding of longer sequences, but this in turn increases execution time and memory usage.
# - alsd_max_target_len: The maximum expected target sequence length during beam search. Larger values allow decoding of longer sequences at the expense of execution time and memory.
# 4) tsd: Time synchronous decoding. Please refer to the paper: Alignment-Length Synchronous Decoding for RNN Transducer for details on the algorithm implemented.
# Time synchronous decoding (TSD) execution time grows by the factor T * max_symmetric_expansions.
# For longer sequences, T is greater and can therefore take a long time for beams to obtain good results. TSD also requires more memory to execute.
# 5) alsd: Alignment-length synchronous decoding. Please refer to the paper: Alignment-Length Synchronous Decoding for RNN Transducer for details on the algorithm implemented.
# Alignment-length synchronous decoding (ALSD) execution time is faster than TSD, with a growth factor of T + U_max, where U_max is the maximum target length expected during execution.
# Generally, T + U_max < T * max_symmetric_expansions. However, ALSD beams are non-unique.
# Therefore it is required to use larger beam sizes to achieve the same (or close to the same) decoding accuracy as TSD.
# For a given decoding accuracy, it is possible to attain faster decoding via ALSD than TSD.

# bh: fused batch step
# model.joint.fuse_loss_wer = True
# model.joint.fused_batch_size = 16  # this can be any value (preferably less than model.*_ds.batch_size)

# bh: hyperparams
# optim.lr=0.01: a lower LR to preserve the pre-trained weights of the encoder; can be the same when freezing encoder; different LRs for enc and dec
# Pick your warmup and LR scaler such that your LR doesn't drop below 1e-5 at worst. Best to keep it around 5e-5 or slightly higher.
# model.optim.sched.warmup_steps=2000 / model.optim.sched.warmup_ratio=0.05: reduced warmup for little finetuning dataset
# For low resource languages, it might be better to increase augmentation via SpecAugment to reduce overfitting. However, this might, in turn, make it too hard for the model to train in a short number of epochs.
# model.spec_augment.time_masks=2: If the average audio durations is less than 10 seconds, you want to reduce spec augment time masks to 2 instead of 10.

# bh: options
# model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm.json" \

# model.optim.betas=[0.9,0.999] \

# ? umbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy
# ? noise on the fly
# ? numba rnnt loss, fastemit regularization
# ? config.model.model_defaults.filters = 128  # Use just 128 filters across the model to speed up training and reduce parameter count
# ?   decoding_config.preserve_alignments = True, decoding_config.fused_batch_size = -1  # temporarily stop fused batch during inference.
# ? exp_manager.checkpoint_callback_params.save_best_model=True \

# todo: stochastic_depth_drop_prob, dropout too high?, multiple gpus

# bh: online augmentation
# +model.train_ds.augmentor.white_noise.prob=1.0 \
# +model.train_ds.augmentor.white_noise.min_level=-90 \
# +model.train_ds.augmentor.white_noise.max_level=-46 \
# +model.train_ds.augmentor.shift.prob=1.0 \
# +model.train_ds.augmentor.shift.min_shift_ms=-5.0 \
# +model.train_ds.augmentor.shift.max_shift_ms=5.0 \

# model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json" \
# model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw_wo_space_after_apostrophe.json" \
# model.train_ds.manifest_filepath="/home/bhuang/asr/NeMo/examples/asr/data/train_hmhm_merged_and_raw_wo_space_after_apostrophe_augmented/train_hmhm_merged_and_raw_wo_space_after_apostrophe_augmented_and_raw.json" \

# todo
# +init_from_nemo_model.model.path="/home/bhuang/.cache/torch/NeMo/NeMo_1.13.0rc0/stt_fr_conformer_transducer_large/0afcc58c13c5341db452f7a37e5ee0bd/stt_fr_conformer_transducer_large.nemo" \

    # exp_manager.resume_if_exists=True \

# train
python ${NEMO_GIT_FOLDER}/examples/asr/asr_transducer/speech_to_text_rnnt_bpe_b.py \
    --config-path="../conf/conformer" --config-name="conformer_transducer_bpe" \
    name="stt_fr_conformer_transducer_large" \
    +init_from_pretrained_model="stt_fr_conformer_transducer_large" \
    model.train_ds.manifest_filepath="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw_wo_space_after_apostrophe.json" \
    model.train_ds.batch_size=8 \
    model.train_ds.use_start_end_token=True \
    model.train_ds.trim_silence=True \
    model.train_ds.max_duration=30 \
    model.train_ds.min_duration=1 \
    +model.train_ds.augmentor.white_noise.prob=1.0 \
    +model.train_ds.augmentor.white_noise.min_level=-90 \
    +model.train_ds.augmentor.white_noise.max_level=-46 \
    +model.train_ds.augmentor.shift.prob=1.0 \
    +model.train_ds.augmentor.shift.min_shift_ms=-5.0 \
    +model.train_ds.augmentor.shift.max_shift_ms=5.0 \
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
    model.encoder.dropout=0.05 \
    model.encoder.dropout_pre_encoder=0.05 \
    model.encoder.dropout_att=0.05 \
    model.decoder.prednet.dropout=0.05 \
    model.joint.jointnet.dropout=0.05 \
    model.joint.fused_batch_size=8 \
    model.optim.name="adamw" \
    model.optim.lr=0.0001 \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.name="CosineAnnealing" \
    ~model.optim.sched.d_model \
    model.optim.sched.warmup_steps=2000 \
    model.optim.sched.min_lr=1e-6 \
    model.spec_augment.time_masks=2 \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    ~trainer.strategy \
    trainer.accumulate_grad_batches=32 \
    trainer.precision=16 \
    trainer.max_epochs=32 \
    trainer.val_check_interval=0.5 \
    +exp_manager.explicit_log_dir="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_wo_space_after_apostrophe_augmented_and_raw_ft_pretrained_bpe_lowerdrp" \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="conformer-transducer-hmhm_merged_and_raw_wo_space_after_apostrophe_augmented_and_raw-ft-pretrained_bpe_lowerdrp" \
    exp_manager.wandb_logger_kwargs.project="nemo-asr-hmhm"


# average checkpoints to .nemo
# python ${NEMO_GIT_FOLDER}/scripts/checkpoint_averaging/checkpoint_averaging.py \
#     examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/checkpoints/stt_fr_conformer_transducer_large.nemo
#     --class_path nemo.collections.asr.models.EncDecRNNTBPEModel
