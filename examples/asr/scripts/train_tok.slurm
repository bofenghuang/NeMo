#!/bin/bash
# Copyright 2024  Bofeng Huang

#SBATCH --job-name=train-tok
#SBATCH --output=logs/%x/%j.out      # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err       # error file (%j = job ID)

##SBATCH --nodes=1
##SBATCH --exclusive
##SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
##SBATCH --cpus-per-task=96           # number of cores per tasks
##SBATCH --gres=gpu:4                 # reserve number of GPUs per node
##SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
##SBATCH --hint=nomultithread         # we get physical cores not logical
##SBATCH --constraint=h100
##SBATCH --account=gkb@h100           # account to use, with xyz the 3 letter code of your project

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_cpu-t3             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --account=cjc@cpu            # cpu accounting

# Train tokenizer

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

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

export HYDRA_FULL_ERROR=1

# NEMO_GIT_FOLDER="/home/bhuang/NeMo"
NEMO_GIT_FOLDER="$HOME/NeMo"

# arg
lang=$1

output_dir=

input_root=/lustre/fsn1/projects/rech/gkb/commun/corpus/speech/stt-pseudo-labeled-whisper-large-v3-multilingual

if [ "$lang" = "it" ]; then
    # it
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/it/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/italian/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/facebook/voxpopuli/it/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/multilingual-tedx/it-it/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/it000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/it100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/it101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "fr" ]; then
    # fr
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/fr/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/french/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/facebook/voxpopuli/fr/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/multilingual-tedx/fr-fr/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/fr000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/fr100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/fr101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/fr102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/fr103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "es" ]; then
    # es
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/es/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/spanish/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/facebook/voxpopuli/es/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/multilingual-tedx/es-es/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/es000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/es100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/es101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "pt" ]; then
    # pt
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/pt/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/portuguese/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/multilingual-tedx/pt-pt/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/pt000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/pt100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/pt101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/pt102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/pt103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "de" ]; then
    # de
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/de/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/german/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/facebook/voxpopuli/de/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/multilingual-tedx/de-de/train_concatenated/train_mtedx_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/de000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/de100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/de101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/de102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "nl" ]; then
    # nl
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/nl/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_jz_pnc.json
        $input_root/facebook/multilingual_librispeech/dutch/train_concatenated/train_facebook_multilingual_librispeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/facebook/voxpopuli/nl/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/nl000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
        $input_root/espnet/yodas/nl100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_jz_pnc.json
    )
elif [ "$lang" = "en" ]; then
    # en
    input_files=(
        $input_root/mozilla-foundation/common_voice_17_0/en/train_concatenated/train_mozilla-foundation_common_voice_17_0_manifest_whisper_large_v3_norm_wer_filt_wer_zipped_pnc.json
        $input_root/openslr/librispeech_asr/train_concatenated/train_openslr_librispeech_asr_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/facebook/voxpopuli/en/train_concatenated/train_facebook_voxpopuli_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/LIUM/tedlium/release3/train_concatenated/train_LIUM_tedlium_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/MLCommons/peoples_speech/default/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/MLCommons/peoples_speech/clean_sa/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/MLCommons/peoples_speech/dirty/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/MLCommons/peoples_speech/dirty_sa/train_concatenated/train_MLCommons_peoples_speech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/speechcolab/gigaspeech/l/train_concatenated/train_speechcolab_gigaspeech_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/distil-whisper/ami-ihm/ihm/train_concatenated/train_distil-whisper_ami-ihm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/distil-whisper/ami-sdm/sdm/train_concatenated/train_distil-whisper_ami-sdm_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en000/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en001/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en002/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en003/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en004/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en005/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en100/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en101/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en102/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en103/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en104/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en105/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en106/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en107/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en109/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en110/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en111/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en112/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en113/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en114/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en115/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en116/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en117/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en118/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en119/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en120/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en121/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en122/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en123/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en124/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en125/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en126/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
        $input_root/espnet/yodas/en127/train_concatenated/train_espnet_yodas_manifest_whisper_large_v3_norm_upprev_wer_filt_zipped_pnc.json
    )
fi

input_manifest_filepath=$(IFS=,; echo "${input_files[*]}")

# tokenize cumstomized dataset
python ${NEMO_GIT_FOLDER}/scripts/tokenizers/process_asr_text_tokenizer_b.py \
    --manifest="${input_manifest_filepath}" \
    --data_root="${output_dir}/${lang}" \
    --vocab_size=1024 \
    --tokenizer="spe" \
    --spe_type="bpe" \
    --spe_character_coverage=0.9995 \
    --spe_remove_extra_whitespaces \
    --no_lower_case \
    --log

