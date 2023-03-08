#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(
    config_path="../conf/conformer/hybrid_transducer_ctc/", config_name="conformer_hybrid_transducer_ctc_bpe"
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)
    print(asr_model.summarize())
    print(asr_model.cfg)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
