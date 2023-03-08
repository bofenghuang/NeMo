#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Modified from: scripts/dataset_processing/add_noise.py

USAGE:
./prep_add_noise_offline.py \
    --input_manifest /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_wo_space_after_apostrophe.json \
    --noise_manifest /home/bhuang/asr/NeMo/examples/asr/data/musan/musan_wo_speech.json \
    --output_manifest /home/bhuang/asr/NeMo/examples/asr/data/train_hmhm_wo_space_after_apostrophe_augmented/train_hmhm_wo_space_after_apostrophe_augmented.json \
    --out_dir /home/bhuang/asr/NeMo/examples/asr/data/train_hmhm_wo_space_after_apostrophe_augmented/audios \
    --num_workers 8

To be able to reproduce the same noisy dataset, use a fixed seed and num_workers=1
"""


import argparse
import copy
import json
import multiprocessing
import os
import random

import numpy as np
import soundfile as sf

from nemo.collections.asr.parts.preprocessing import perturb, segment

rng = None
att_factor = 0.8
sample_rate = 16000

perturber = None


def create_manifest(input_manifest, output_manifest, out_dir):
    output_dir = os.path.dirname(output_manifest)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_manifest, "r") as inf, open(output_manifest, "w") as outf:
        for line in inf:
            row = json.loads(line.strip())
            row["audio_filepath"] = os.path.join(out_dir, os.path.basename(row["audio_filepath"]))
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_row(row):
    audio_file = row["audio_filepath"]
    out_dir = row["out_dir"]

    # todo: timestamp
    data = segment.AudioSegment.from_file(audio_file, target_sr=sample_rate, offset=0)

    # data = copy.deepcopy(data_orig)
    perturber.perturb(data)

    # todo : norm?
    # max_level = np.max(np.abs(data.samples))
    # norm_factor = att_factor / max_level
    # new_samples = norm_factor * data.samples

    os.makedirs(out_dir, exist_ok=True)
    out_f = os.path.join(out_dir, os.path.basename(audio_file))
    sf.write(out_f, data.samples, sample_rate)


def add_noise(infile, out_dir, num_workers=1):
    allrows = []

    with open(infile, "r") as inf:
        for line in inf:
            row = json.loads(line.strip())
            row["out_dir"] = out_dir
            allrows.append(row)
    pool = multiprocessing.Pool(num_workers)
    pool.map(process_row, allrows)
    pool.close()
    print("Done!")


# +model.train_ds.augmentor.white_noise.prob=0.2 \
# +model.train_ds.augmentor.white_noise.min_level=-50 \
# +model.train_ds.augmentor.white_noise.max_level=-10 \
# +model.train_ds.augmentor.shift.prob=0.2 \
# +model.train_ds.augmentor.shift.min_shift_ms=-5.0 \
# +model.train_ds.augmentor.shift.max_shift_ms=5.0 \


def init_perturber(noise_manifest_path, rir_manifest_path=None):
    perturbations = [
        perturb.TimeStretchPerturbation(min_speed_rate=0.9, max_speed_rate=1.1, num_rates=3),
        perturb.NoisePerturbation(manifest_path=noise_manifest_path, min_snr_db=0, max_snr_db=50, max_gain_db=300.0, rng=rng),
        # perturb.RirAndNoisePerturbation(
        #     rir_manifest_path=rir_manifest_path,
        #     rir_prob=1,
        #     noise_manifest_paths=[noise_manifest_path],  # use noise_manifest_path from previous step
        #     noise_tar_filepaths=[None],  # `[None]` to indicates that noise audio files are not tar.
        #     min_snr_db=[10],  # foreground noise snr
        #     max_snr_db=[30],
        #     bg_noise_manifest_paths=[noise_manifest_path],
        #     bg_noise_tar_filepaths=[None],
        #     bg_min_snr_db=[10],  # background noise snr
        #     bg_max_snr_db=[30],
        # )
    ]

    # ! to customize
    probas = [0.2, 0.2]

    return perturb.AudioAugmentor(list(zip(probas, perturbations)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="clean test set",
    )
    parser.add_argument("--noise_manifest", type=str, required=True, help="path to noise manifest file")
    # parser.add_argument("--rir_manifest", type=str, required=True, help="path to rir manifest file")
    parser.add_argument("--output_manifest", type=str, required=True, help="destination directory for manifests")
    parser.add_argument("--out_dir", type=str, required=True, help="destination directory for audio and manifests")
    # parser.add_argument("--snrs", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument(
        "--attenuation_factor",
        default=0.8,
        type=float,
        help="Attenuation factor applied on the noise added samples before writing to wave",
    )
    args = parser.parse_args()
    global sample_rate
    sample_rate = args.sample_rate
    global rng
    rng = args.seed
    num_workers = args.num_workers

    global perturber
    perturber = init_perturber(
        noise_manifest_path=args.noise_manifest,
        # rir_manifest_path=args.rir_manifest,
    )

    add_noise(args.input_manifest, args.out_dir, num_workers=num_workers)
    create_manifest(args.input_manifest, args.output_manifest, args.out_dir)


if __name__ == "__main__":
    main()
