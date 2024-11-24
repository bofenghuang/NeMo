#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

"""
Normalize text column in manifest.

USAGE:
python prep_normalize_text.py /projects/bhuang/corpus/speech/nemo_manifests/media_speech/FR/media_speech_manifest.json
"""

import sys

sys.path.append("/home/bhuang/myscripts")

import re

import fire
from datasets import load_dataset
from hf_dataset_processing.file_utils import write_dataset_to_json
from text_normalization.normalize_french import FrenchTextNormalizer


def main(input_file, num_proc=32):

    path, ext = input_file.rsplit(".", 1)
    output_file = f"{path}_normalized.{ext}"

    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    # print(raw_dataset)

    # debug
    # raw_dataset = raw_dataset.select(range(10))

    # "nan" in text parsed into None
    raw_dataset = raw_dataset.filter(
        lambda x: x["text"] is not None,
        num_proc=num_proc,
    )

    normalizer = FrenchTextNormalizer()

    def process_function(example):
        s = example["text"]

        # apostrophe
        # s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe
        # s = re.sub(r"'\s+", "'", s)  # standardize when there's a space after an apostrophe
        # s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space

        # patch for mtedx
        # s = re.sub(r"\bapplaudissements\b", "", s)
        # (Rires et applaudissements) La sexualité des vieux
        # s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        # s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis

        s = normalizer(s, do_lowercase=True, do_ignore_words=False, symbols_to_keep="'-", do_standardize_numbers=True)

        example["text"] = s

        return example

    processed_dataset = raw_dataset.map(
        process_function,
        num_proc=num_proc,
        # remove_columns=raw_datasets.column_names,
        load_from_cache_file=False,
        desc="process",
    )
    # print(processed_dataset)

    # processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    write_dataset_to_json(processed_dataset, output_file)
    print(f"The processed data is saved into {output_file}")

    # with open(input_file, "r") as fin, open(output_file, "w") as fout:
    #     for line in fin:
    #         line = line.rstrip("\n")
    #         line_data = json.loads(line)
    #         # print(line_data)
    #         line_data["text"] = normalize_text(line_data["text"])
    #         # print(line_data)
    #         # quit()

    #         json.dump(line_data, fout, ensure_ascii=False)
    #         fout.write("\n")


if __name__ == "__main__":
    # main()  # noqa pylint: disable=no-value-for-parameter
    fire.Fire(main)
