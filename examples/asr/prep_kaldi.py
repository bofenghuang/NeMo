#!/usr/bin/env python
# coding=utf-8

""""
nemo json manifest to kaldi
"""

import json
import sys
import csv
import os
import pandas as pd
from tqdm import tqdm
import subprocess

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)


def main():
    input_csv_path = sys.argv[1]
    out_dir = sys.argv[2]

    data = []
    with open(input_csv_path) as f:
        for line in f:
            line = line.rstrip()
            data.append(json.loads(line))

    df_data = pd.DataFrame(data)
    # df_data["utt"] = df_data["audio_filepath"].str.extract(r"\/([0-9a-zA-Z\_\-]+)\.wav")
    df_data["utt"] = df_data["audio_filepath"].str.rsplit("/", n=2).str[-1]
    df_data["utt"] = df_data["utt"].str.slice(0, -4)
    df_data["spk"] = df_data["utt"].str.slice(0, -4)
    df_data.sort_values(by="utt", inplace=True)
    print(df_data.head())

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_data[["utt", "text"]].to_csv(
        f"{out_dir}/text", sep="\t", quoting=csv.QUOTE_NONE, index=False, header=False
    )
    subprocess.run(f"sed -i \"s/\t/ /g\" {out_dir}/text", shell=True)

    df_data[["utt", "spk"]].to_csv(
        f"{out_dir}/utt2spk", sep=" ", quoting=csv.QUOTE_NONE, index=False, header=False
    )

    df_data[["utt", "audio_filepath"]].to_csv(
        f"{out_dir}/wav.scp", sep=" ", quoting=csv.QUOTE_NONE, index=False, header=False
    )


if __name__ == "__main__":
    main()
