#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import sys

sys.path.append("/home/bhuang/myscripts")

import re
from collections import defaultdict

import fire
import string
import numpy as np
from datasets import load_dataset
from hf_dataset_processing.file_utils import write_dataset_to_json


def main(input_file_path, output_file_path, min_duration_s=0.1, max_duration_s=30, max_identical_text=5, num_proc=32):
    dataset = load_dataset("json", data_files=input_file_path)["train"]
    print(f'Initial load: {len(dataset):,d} utterances, {sum(dataset["duration"]) / 3600:.2f}h')

    # debug
    # dataset = dataset.select(range(50000))

    # remove utt with empty text
    dataset = dataset.filter(lambda x: x["text"] is not None and x["text"], num_proc=num_proc)
    # at least one alphabet
    dataset = dataset.filter(lambda x: bool(re.search(rf"[\w]", x["text"])), num_proc=num_proc)
    print(f'Remove empty text: {len(dataset):,d} utterances, {sum(dataset["duration"]) / 3600:.2f}h')

    # still got number: number like "MP4" can't be converted
    # dataset = dataset.filter(
    #     lambda x: not bool(re.search("[0-9]", x["text"])),
    #     num_proc=num_proc,
    # )
    # print(f'Heuristic: {len(dataset):,d} utterances, {sum(dataset["duration"]) / 3600:.2f}h')

    # remove short and long utt
    dataset = dataset.filter(
        lambda x: x["duration"] > min_duration_s and x["duration"] < max_duration_s, num_proc=num_proc,
    )
    print(f'Remove short and long: {len(dataset):,d} utterances, {sum(dataset["duration"]) / 3600:.2f}h')

    # dedup text
    # https://github.com/huggingface/datasets/issues/2514
    def get_hash(example):
        """Get hash of content field."""
        return {"hash": hash(example["text"])}  # can use any hashing function here

    # def check_uniques(example, uniques):
    #     """Check if current hash is still in set of unique hashes and remove if true."""
    #     if example["hash"] in uniques:
    #         uniques.remove(example["hash"])
    #         return True
    #     else:
    #         return False

    def check_uniques(example, text_counts):
        text_counts[example["hash"]] += 1
        if text_counts[example["hash"]] <= max_identical_text:
            return True
        else:
            return False

    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(get_hash, num_proc=num_proc)
    # dedup
    # uniques = set(dataset.unique("hash"))
    # dataset = dataset.filter(check_uniques, fn_kwargs={"uniques": uniques})
    # dedup by max number
    text_counts = defaultdict(int)
    dataset = dataset.filter(check_uniques, fn_kwargs={"text_counts": text_counts})
    dataset = dataset.remove_columns("hash")
    dataset = dataset.sort(["split", "audio_filepath"])
    print(f'Dedup by text: {len(dataset):,d} utterances, {sum(dataset["duration"]) / 3600:.2f}h')

    write_dataset_to_json(dataset, output_file_path)
    print(f"The preprocessed data is saved into {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
