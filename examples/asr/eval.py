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

# from myscripts.text.normalize_french import FrenchTextNormalizer
from myscripts.text.normalize_french_zaion import FrenchTextNormalizer
from myscripts.text.compute_wer import compute_wer


def main(dataset_manifest_path, out_dir=None, do_ignore_words=False):
    data = []
    with open(dataset_manifest_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    result_df = pd.DataFrame(data)
    result_df.rename(columns={"audio_filepath": "wav", "text": "wrd", "pred_text": "pred_wrd"}, inplace=True)
    result_df["ID"] = result_df["wav"].map(lambda x: Path(x).stem)
    print(result_df.head())
    print(f"#utterances: {result_df.shape[0]}")

    text_normalizer = FrenchTextNormalizer()

    target_column_name = "wrd"
    prediction_column_name = "pred_wrd"
    id_column_name = "ID"

    def norm_func(s):
        # NB
        return text_normalizer(
            s, do_lowercase=True, do_ignore_words=do_ignore_words, symbols_to_keep="'-", do_standardize_numbers=True
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
