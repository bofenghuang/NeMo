#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

""""
Convert speechbrain csv to nemo json manifest and upsample & save audio segments

Only for Zaion data with a sample rate of 8000

USAGE:
./prep_csv_to_manifest.py \
    --input_file_path /home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv \
    --manifest_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
    --out_audio_dir /home/bhuang/corpus/speech/internal/hm_hm_16k/audios_16k/train_hmhm_merged_and_raw
"""

import json
import os
import subprocess
import sys

import fire
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def load_resample_write(in_wav_path, out_wav_path, in_sr, out_sr, offset, duration):
    # cmd = ["sox", ]
    # subprocess.run(cmd)

    y, _ = librosa.load(in_wav_path, sr=in_sr, offset=offset, duration=duration)
    y_upsampled = librosa.resample(y, orig_sr=in_sr, target_sr=out_sr)
    # obsolete
    # librosa.output.write_wav(out_wav_path, y_upsampled, out_sr)
    # sf.write(out_wav_path, y_upsampled, out_sr, "PCM_16")
    sf.write(out_wav_path, y_upsampled, out_sr)


def build_manifest(input_file_path, manifest_path, out_audio_dir=None):

    if out_audio_dir is not None and not os.path.exists(out_audio_dir):
        os.makedirs(out_audio_dir)

    manifest_dir = os.path.dirname(manifest_path)
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)

    df = pd.read_csv(input_file_path)
    # bh: nan in dekuple
    # df = pd.read_csv(input_file_path, keep_default_na=False)
    # df.columns = ["ID", "start", "end", "duration", "wrd", "wav"]
    # print(df[df["wrd"].isna()])
    # quit()

    # todo: np vectorization
    df_dict = df.to_dict("records")
    with open(manifest_path, "w") as fout:
        # for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        for row in tqdm(df_dict):

            if out_audio_dir is None:
                # Write the metadata to the manifest
                # sample to second
                metadata = {
                    "audio_filepath": row["wav"],
                    "offset": row["start"] / 8000,
                    "duration": row["duration"] / 8000,
                    "text": row["wrd"],
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write("\n")
            else:
                out_wav_path = os.path.join(out_audio_dir, f'{row["ID"]}.wav')

                # ! comment if necessary
                load_resample_write(
                    in_wav_path=row["wav"],
                    out_wav_path=out_wav_path,
                    in_sr=8_000,
                    out_sr=16_000,
                    # offset=row["start"] / 8000,
                    # duration=row["duration"] / 8000,
                    offset=row["start"],
                    duration=row["duration"],
                )

                # Write the metadata to the manifest
                # sample to second
                metadata = {
                    "audio_filepath": out_wav_path,
                    # "duration": row["duration"] / 8000,
                    "duration": row["duration"],
                    "text": row["wrd"],
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write("\n")

            # debug
            # quit()


if __name__ == "__main__":
    fire.Fire(build_manifest)
