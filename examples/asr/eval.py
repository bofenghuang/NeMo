#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang


import json
import os
import re
import sys
from pathlib import Path

import fire
import pandas as pd

# NB
from text_normalization.normalize_french import FrenchTextNormalizer
# from text_normalization.normalize_french_zaion import FrenchTextNormalizer
from asr_metric_calculation.compute_wer import compute_wer


def main(
    dataset_manifest_path,
    out_dir=None,
    audio_column_name="audio_filepath",
    target_column_name="text",
    prediction_column_name="pred_text",
    id_column_name="ID",
    do_lowercase=True,
    do_ignore_words=False,
    symbols_to_keep="'-",
    do_standardize_numbers=True,
):
    data = []
    with open(dataset_manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            # bh: for dekuple
            # line = re.sub(r"\bNaN\b", '"nan"', line)
            data.append(json.loads(line))

    result_df = pd.DataFrame(data)
    # result_df.rename(columns={"audio_filepath": "wav", "text": "wrd", "pred_text": "pred_wrd"}, inplace=True)
    result_df[id_column_name] = result_df[audio_column_name].map(lambda x: Path(x).stem)
    # print(result_df.head())
    print(f"#utterances: {result_df.shape[0]}")

    text_normalizer = FrenchTextNormalizer()
    # text_normalizer = FrenchTextNormalizer(remove_diacritics=True)

    # print(result_df[result_df[target_column_name].isna()])
    # quit()

    def norm_func(s):
        # NB
        return text_normalizer(
            s,
            do_lowercase=do_lowercase,
            do_ignore_words=do_ignore_words,
            symbols_to_keep=symbols_to_keep,
            do_standardize_numbers=do_standardize_numbers,
        )

    result_df[target_column_name] = result_df[target_column_name].map(norm_func)
    result_df[prediction_column_name] = result_df[prediction_column_name].map(norm_func)

    # filtering out empty targets
    result_df = result_df[result_df[target_column_name] != ""]
    print(f"#utterances (after filtering): {result_df.shape[0]}")

    result_df[target_column_name] = result_df[target_column_name].str.split()
    targets = result_df.set_index(id_column_name)[target_column_name].to_dict()

    result_df[prediction_column_name] = result_df[prediction_column_name].str.split()
    predictions = result_df.set_index(id_column_name)[prediction_column_name].to_dict()

    out_dir_ = f"{out_dir}/wer_summary_without_fillers" if do_ignore_words else f"{out_dir}/wer_summary"
    compute_wer(targets, predictions, out_dir_, do_print_top_wer=True, do_catastrophic=True)


if __name__ == "__main__":
    fire.Fire(main)
