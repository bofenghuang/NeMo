#!/bin/bash
# Copyright 2023  Bofeng Huang

#SBATCH --job-name=tr
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --gres=gpu:4                 # reserve number of GPUs per node
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=h100
#SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project

# https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html
# Train fastconformer_hybrid_transducer_ctc_bpe model from scratch on public data

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load arch/h100
module load git-lfs
# module load unrar
module load anaconda-py3/2024.06
module load cuda/12.2.0
# conda activate speech
# conda activate asr
conda activate nemo

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# slurm buffers python output by default, unbuffer python -u
export PYTHONUNBUFFERED="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_EVALUATE_OFFLINE="1"

# wandb
export WANDB_MODE=offline
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
# export WANDB_PROJECT=hf-whisper-v4.1
# export WANDB_PROJECT=hf-whisper-v4.2
# export WANDB_PROJECT=hf-whisper-v4.3
export WANDB_PROJECT=hf-whisper-distil-multi-v0.3

# hydra
export HYDRA_FULL_ERROR="1"

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="2,3"

# NEMO_GIT_FOLDER="/home/bhuang/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

# data
# input_root=/projects/bhuang/corpus/speech/nemo_manifests
input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

train_files=(
    # $input_root/final_nemo/train_manifest_it_wer20.json
    $input_root/final_nemo/train_manifest_fr_wer20.json
    $input_root/final_nemo/train_manifest_es_wer20.json
    # $input_root/final_nemo/train_manifest_pt_wer20.json
    $input_root/final_nemo/train_manifest_de_wer20.json
    # $input_root/final_nemo/train_manifest_nl_wer20.json
    # $input_root/final_nemo/train_manifest_en_wer10.json
    $input_root/final_nemo/train_manifest_en_wer10_subset30kh.json
)

# validation_files=(
#     # $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     # $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     # $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm.json
#     # $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest_nm.json
#     $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest_nm.json
#     $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest_nm.json
#     # $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest_nm.json
#     $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest_nm.json
#     # $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest_nm.json
#     $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest_nm.json
# )

validation_files=(
    # model selection based on 1st validation result
    # $input_root/google/fleurs/it_it/validation/validation_google_fleurs_manifest_nm_it_it.json
    $input_root/google/fleurs/fr_fr/validation/validation_google_fleurs_manifest_nm_fr_fr.json
    $input_root/google/fleurs/es_419/validation/validation_google_fleurs_manifest_nm_es_419.json
    # $input_root/google/fleurs/pt_br/validation/validation_google_fleurs_manifest_nm_pt_br.json
    $input_root/google/fleurs/de_de/validation/validation_google_fleurs_manifest_nm_de_de.json
    # $input_root/google/fleurs/nl_nl/validation/validation_google_fleurs_manifest_nm_nl_nl.json
    $input_root/google/fleurs/en_us/validation/validation_google_fleurs_manifest_nm_en_us.json
    # $input_root/mozilla-foundation/common_voice_17_0/it/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_it.json
    $input_root/mozilla-foundation/common_voice_17_0/fr/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_fr.json
    $input_root/mozilla-foundation/common_voice_17_0/es/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_es.json
    # $input_root/mozilla-foundation/common_voice_17_0/pt/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_pt.json
    $input_root/mozilla-foundation/common_voice_17_0/de/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_de.json
    # $input_root/mozilla-foundation/common_voice_17_0/nl/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_nl.json
    $input_root/mozilla-foundation/common_voice_17_0/en/validation/validation_mozilla-foundation_common_voice_17_0_manifest_nm_en.json
)


# train_manifest_filepath=$(IFS=,; echo "${train_files[*]}")
train_manifest_filepath="[$(IFS=,; echo "${train_files[*]}")]"
# validation_manifest_filepath=$(IFS=,; echo "${validation_files[*]}")
validation_manifest_filepath="[$(IFS=,; echo "${validation_files[*]}")]"

# tokenizer
# tokenizer_dir="${NEMO_GIT_FOLDER}/examples/asr/nemo_experiments/tokenizers/tokenizer_spe_bpe_v1024"
tokenizer_root="/lustre/fswork/projects/rech/gkb/commun/outputs/nemo_experiments/tokenizers"

# model
init_model="/lustre/fswork/projects/rech/gkb/commun/models/pretrained/nemo/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo"

# output
output_root="/lustre/fswork/projects/rech/gkb/commun/outputs/nemo_experiments/stt"
outdir="${output_root}/base_m4l_ep100_bs4096_lr068"

# wandb
wandb_name="${outdir##*/}"

# Set CPUs
num_workers=64
# Set GPUs
num_nodes="${SLURM_NNODES}"

    # tokenizer
    # model.tokenizer.type="bpe" \
    # model.tokenizer.dir="$tokenizer_dir" \
    # https://github.com/NVIDIA/NeMo/issues/6585
    # https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Multilang_ASR.ipynb
    # model.tokenizer.type="agg" \
    # +model.tokenizer.langs.it.type="bpe" \
    # +model.tokenizer.langs.it.dir="${tokenizer_root}/it/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.fr.type="bpe" \
    # +model.tokenizer.langs.fr.dir="${tokenizer_root}/fr/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.es.type="bpe" \
    # +model.tokenizer.langs.es.dir="${tokenizer_root}/es/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.pt.type="bpe" \
    # +model.tokenizer.langs.pt.dir="${tokenizer_root}/pt/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.de.type="bpe" \
    # +model.tokenizer.langs.de.dir="${tokenizer_root}/de/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.nl.type="bpe" \
    # +model.tokenizer.langs.nl.dir="${tokenizer_root}/nl/tokenizer_spe_bpe_v1024" \
    # +model.tokenizer.langs.en.type="bpe" \
    # +model.tokenizer.langs.en.dir="${tokenizer_root}/en/tokenizer_spe_bpe_v1024" \

    # data augmentation
    # https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_Noise_Augmentation.ipynb
    # +model.train_ds.augmentor.white_noise.prob=1.0 \
    # +model.train_ds.augmentor.white_noise.min_level=-90 \
    # +model.train_ds.augmentor.white_noise.max_level=-46 \
    # +model.train_ds.augmentor.shift.prob=1.0 \
    # +model.train_ds.augmentor.shift.min_shift_ms=-5.0 \
    # +model.train_ds.augmentor.shift.max_shift_ms=5.0 \

    # optim
    # model.optim.sched.name="CosineAnnealing" \
    # ~model.optim.sched.d_model \
    # model.optim.sched.min_lr=0.0001 \
    # 
    # ~model.optim.sched.warmup_steps \
    # model.optim.sched.warmup_ratio=0.05 \

    # hparams
    # trainer.enable_checkpointing=True \


srun python ${NEMO_GIT_FOLDER}/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    --config-path="../conf/fastconformer/hybrid_transducer_ctc" --config-name="fastconformer_hybrid_transducer_ctc_bpe" \
    name="stt_fastconformer_hybrid_transducer_ctc_bpe_large" \
    +init_from_nemo_model.model.path="$init_model" \
    +init_from_nemo_model.model.include=["encoder"] \
    model.train_ds.manifest_filepath="$train_manifest_filepath" \
    model.train_ds.batch_size=16 \
    model.train_ds.num_workers=$num_workers \
    model.train_ds.max_duration=30 \
    model.train_ds.min_duration=1 \
    model.validation_ds.manifest_filepath="$validation_manifest_filepath" \
    model.validation_ds.batch_size=16 \
    model.validation_ds.num_workers=$num_workers \
    +model.validation_ds.max_duration=30 \
    +model.validation_ds.min_duration=1 \
    +model.validation_ds.max_utts=10000 \
    ~model.test_ds \
    model.tokenizer.type="agg" \
    +model.tokenizer.langs.fr.type="bpe" \
    +model.tokenizer.langs.fr.dir="${tokenizer_root}/fr/tokenizer_spe_bpe_v1024" \
    +model.tokenizer.langs.es.type="bpe" \
    +model.tokenizer.langs.es.dir="${tokenizer_root}/es/tokenizer_spe_bpe_v1024" \
    +model.tokenizer.langs.de.type="bpe" \
    +model.tokenizer.langs.de.dir="${tokenizer_root}/de/tokenizer_spe_bpe_v1024" \
    +model.tokenizer.langs.en.type="bpe" \
    +model.tokenizer.langs.en.dir="${tokenizer_root}/en/tokenizer_spe_bpe_v1024" \
    model.optim.lr=0.68 \
    model.optim.weight_decay=1e-3 \
    model.optim.sched.name="NoamAnnealing" \
    trainer.num_nodes="${num_nodes}" \
    trainer.max_epochs=100 \
    trainer.val_check_interval=0.25 \
    trainer.accelerator="gpu" \
    trainer.accumulate_grad_batches=4 \
    trainer.precision="bf16-mixed" \
    exp_manager.exp_dir="$outdir" \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="$wandb_name"

echo "END TIME: $(date)"
