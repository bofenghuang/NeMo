#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import re

import fire
import numpy as np
from datasets import load_dataset


def main(input_file):

    raw_dataset = load_dataset("json", data_files=input_file)["train"]
    # print(raw_dataset)

    durations = np.array(raw_dataset["duration"])

    print("\n\n")
    print(f"{input_file.rsplit('/', 1)[-1]}")
    print()
    print(f"Num of UTTs : {len(raw_dataset):,d}")
    print(f"Tot Duration: {durations.sum() / 3600:.2f}h")
    print(f"Avg Duration: {durations.mean():.2f}s")
    print(f"Max Duration: {durations.max():.2f}s")
    print(f"Min Duration: {durations.min():.2f}s")


if __name__ == "__main__":
    fire.Fire(main)
