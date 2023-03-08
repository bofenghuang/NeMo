#!/usr/bin/env python
# coding=utf-8
# Copyright 2021  Bofeng Huang

import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import nemo.collections.asr as nemo_asr


@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    nemo_model_file = "/home/bhuang/asr/NeMo/examples/asr/nemo_experiments/stt_fr_conformer_transducer_large/hmhm_merged_and_raw_ft/checkpoints/stt_fr_conformer_transducer_large.nemo"

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load nemo ptm
    # nemo_asr.models.EncDecCTCModel.list_available_models()
    # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)

    # asr_model = nemo_asr.models.ASRModel.restore_from(nemo_model_file, map_location=device)
    # asr_model = EncDecRNNTBPEModel.restore_from(nemo_model_file, map_location=device)
    asr_model = EncDecRNNTBPEModel.restore_from(nemo_model_file, map_location="cpu")
    print(asr_model.summarize())

    # finally, update the model's internal config
    asr_model.cfg = cfg.model

    asr_model.setup_multiple_test_data(cfg.model.test_ds)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    # Setup model with the trainer
    asr_model.set_trainer(trainer)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)

    # Convert our audio sample to text
    # files = [AUDIO_FILENAME]
    # transcript = asr_model.transcribe(paths2audio_files=files)[0]
    # print(f'Transcript: "{transcript}"')


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
