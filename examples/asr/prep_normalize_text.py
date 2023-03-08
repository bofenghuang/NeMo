#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Normalize text column in manifest.

USAGE:
python prep_normalize_text.py \
    --in_file_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
    --out_file_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw_new.json
"""

import fire
import json
import re


def normalize_text(s):
    # apostrophe
    s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe
    s = re.sub(r"'\s+", "'", s)  # standardize when there's a space after an apostrophe

    s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space

    return s


def main(in_file_path, out_file_path):
    with open(in_file_path, "r") as fin, open(out_file_path, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            line_data = json.loads(line)
            # print(line_data)
            line_data["text"] = normalize_text(line_data["text"])
            # print(line_data)
            # quit()

            json.dump(line_data, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    # main()  # noqa pylint: disable=no-value-for-parameter
    fire.Fire(main)
