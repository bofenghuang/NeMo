#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED="1"

export HYDRA_FULL_ERROR="1"

export CUDA_VISIBLE_DEVICES="0"

NEMO_GIT_FOLDER="/home/bhuang/asr/NeMo"

infer_and_eval() {
    modelpath=$1
    testmanifest=$2
    outdir=$3
    decodeopts=$4

    # echo $modelpath
    # echo $testmanifest
    # echo $outdir
    # echo $decodeopts

    [ -d $outdir ] || mkdir -p $outdir

    # pretrained_name="nvidia/stt_fr_fastconformer_hybrid_large_pc"
    # audio_dir
    # decoder_type

    # model_path=$modelpath \
    # pretrained_name=$modelpath \

    # infer
    python ${NEMO_GIT_FOLDER}/examples/asr/transcribe_speech.py \
        pretrained_name=$modelpath \
        dataset_manifest=$testmanifest \
        output_filename="$outdir/predictions.json" \
        batch_size=1 \
        num_workers=4 \
        append_pred=False \
        pred_name_postfix="" \
        compute_langs=False \
        cuda=0 \
        amp=True \
        ${decodeopts[@]}

    # eval predictions
    python ${NEMO_GIT_FOLDER}/examples/asr/eval.py --dataset_manifest_path "$outdir/predictions.json" --out_dir $outdir --do_ignore_words False
    # python ${NEMO_GIT_FOLDER}/examples/asr/eval.py --dataset_manifest_path "$outdir/predictions.json" --out_dir $outdir_ --do_ignore_words True
}

# decoding options
# decode_opts=(rnnt_decoding.strategy="beam" rnnt_decoding.beam.beam_size=10)
decode_opts=(decoder_type="ctc")

# model
# nvidia
# model_path="nvidia/stt_fr_conformer_ctc_large"
model_path="nvidia/stt_fr_conformer_transducer_large"
# model_path="nvidia/stt_fr_fastconformer_hybrid_large_pc"
# general
# model_path="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_fastconformer_hybrid_transducer_ctc_bpe/large_bs2048_lr1e3/stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large/2023-09-18_20-33-08/checkpoints/stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large.nemo"
# model_path="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_fastconformer_hybrid_transducer_ctc_bpe/large_bs2048_lr1e3/stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large/2023-09-18_20-33-08/checkpoints/stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large-averaged.nemo"
# zaion hmhm
# model_path="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_new/checkpoints/stt_fr_conformer_transducer_large.nemo"
# model_path="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/checkpoints/stt_fr_conformer_transducer_large.nemo"
# model_path="/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft_pretrained_bpe/checkpoints/stt_fr_conformer_transducer_large-averaged.nemo"

# root_outdir="nemo_experiments/pretrained_stt_fr_conformer_ctc_large/outputs"
root_outdir="nemo_experiments/pretrained_stt_fr_conformer_transducer_large/outputs"
# root_outdir="nemo_experiments/pretrained_stt_fr_fastconformer_hybrid_large_pc/outputs"
# root_outdir="nemo_experiments/stt_fr_fastconformer_hybrid_transducer_ctc_bpe/large_bs2048_lr1e3/stt_fr_fastconformer_hybrid_transducer_ctc_bpe_large/2023-09-18_20-33-08/outputs_averaged"

# general data
########################################################################################################################################################################################################
# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/mozilla-foundation/common_voice_13_0/fr/test/test_mozilla-foundation_common_voice_13_0_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mcv_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mcv_test_ctc_greedy" $decode_opts

# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/test/test_facebook_multilingual_librispeech_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mls_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mls_test_ctc_greedy" $decode_opts

# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/test/test_facebook_voxpopuli_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/voxpopuli_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/voxpopuli_test_ctc_greedy" $decode_opts

# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/google/fleurs/fr_fr/test/test_google_fleurs_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/fleurs_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/fleurs_test_ctc_greedy" $decode_opts

# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/mtedx/fr-fr/test/test_mtedx_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mtedx_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/mtedx_test_ctc_greedy" $decode_opts

# test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/test/test_gigant_african_accented_french_manifest_normalized.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/african_accented_french_test_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/african_accented_french_test_ctc_greedy" $decode_opts

test_manifest="/projects/bhuang/corpus/speech/nemo_manifests/speech-recognition-community-v2/dev_data/fr/validation/validation_speech-recognition-community-v2_dev_data_manifest_normalized.json"
infer_and_eval $model_path $test_manifest "${root_outdir}/speech_recognition_community_v2_dev_data_greedy"
# infer_and_eval $model_path $test_manifest "${root_outdir}/speech_recognition_community_v2_dev_data_ctc_greedy" $decode_opts

# zaion data
########################################################################################################################################################################################################

# test_manifest="/home/ywang/NeMo/examples/asr_zaion/190h_manifest/test_manifest_segment_16k_190h.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_hmhm_greedy"
# # infer_and_eval $model_path $test_manifest "${root_outdir}/test_hmhm_beam10" $decode_opts
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_hmhm_ctc_greedy" $decode_opts

# test_manifest="/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/carglass_5h/16k_segment.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_carglass_greedy"
# # # infer_and_eval $model_path $test_manifest "${root_outdir}/test_carglass_beam10" $decode_opts
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_carglass_ctc_greedy" $decode_opts

# test_manifest="/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/dekuple_5h/16k_segment.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_dekuple_greedy"
# # # infer_and_eval $model_path $test_manifest "${root_outdir}/test_dekuple_beam10" $decode_opts
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_dekuple_ctc_greedy" $decode_opts

# test_manifest="/home/ywang/NeMo/examples/asr_zaion/benchmark_robustess/lbpa_2.35h/16k_segment.json"
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_lbpa_greedy"
# # # infer_and_eval $model_path $test_manifest "${root_outdir}/test_lbpa_beam10" $decode_opts
# infer_and_eval $model_path $test_manifest "${root_outdir}/test_lbpa_ctc_greedy" $decode_opts
