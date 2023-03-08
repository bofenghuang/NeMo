#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang


"""
Lets prepare a manifest file using the baseline file itself, cut into 1 second segments.

USAGE:
python examples/asr/prep_audio_files_to_manifest.py \
    --audio_dir /home/bhuang/corpus/speech/public/musan_wo_speech \
    --manifest_path examples/asr/data/musan.json

"""

import json
import os
from pathlib import Path

import fire
import librosa
from tqdm import tqdm


# todo: resampling, cutoff args


# Modified from: https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Online_Noise_Augmentation.ipynb#scrollTo=wK8uwpt16d6I
def write_manifest(
    audio_dir,
    manifest_path,
    suffix="wav",
    duration_max=None,
    duration_stride=1.0,
    filter_long=False,
    duration_limit=10.0,
):
    manifest_dir = os.path.dirname(manifest_path)
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)

    if duration_max is None:
        duration_max = 1e9

    # with open(os.path.join(data_dir, manifest_name + ".json"), "w") as fout:
    with open(manifest_path, "w") as fout:

        paths = Path(audio_dir).rglob(f"*.{suffix}")
        paths_list = list(paths)

        for p in tqdm(paths_list):
            filepath = p.as_posix()

            try:
                y, _sr = librosa.load(filepath)
                duration = librosa.get_duration(y=y, sr=_sr)

            except Exception:
                print(f"\n>>>>>>>>> WARNING: Librosa failed to load file {filepath}. Skipping this file !\n")
                return

            if filter_long and duration > duration_limit:
                print(f"Skipping sound sample {filepath}, exceeds duration limit of {duration_limit}")
                return

            offsets = []
            durations = []

            if duration > duration_max:
                current_offset = 0.0

                while current_offset < duration:
                    difference = duration - current_offset
                    segment_duration = min(duration_max, difference)

                    offsets.append(current_offset)
                    durations.append(segment_duration)

                    current_offset += duration_stride

            else:
                offsets.append(0.0)
                durations.append(duration)

            for duration, offset in zip(durations, offsets):
                metadata = {
                    "audio_filepath": filepath,
                    "duration": duration,
                    "label": "noise",
                    "text": "_",  # for compatibility with ASRAudioText collection
                    "offset": offset,
                }

                json.dump(metadata, fout)
                fout.write("\n")
                fout.flush()

            # print(f"Wrote {len(durations)} segments for filename {filepath}")

    # print("Finished preparing manifest !")


if __name__ == "__main__":
    fire.Fire(write_manifest)
