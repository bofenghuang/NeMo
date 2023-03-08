#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

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

    # trainer = pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    # asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    # # Initialize the weights of the model from another model, if provided via config
    # asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_fr_conformer_ctc_large", map_location='cpu')
    # asr_model = nemo_asr.models.ASRModel.from_pretrained(cfg.name, map_location="cpu")
    # bh: need to specify model's class, or can't be loaded
    asr_model = EncDecRNNTBPEModel.from_pretrained(cfg.name, map_location="cpu")
    print(asr_model.summarize())
    # debug
    # print(next(asr_model.named_parameters()))

    # Preserve the decoder parameters in case weight matching can be done later
    # pretrained_decoder = asr_model.decoder.state_dict()
    # pretrained_joint = asr_model.joint.state_dict()

    # bh: Update the vocabulary
    # bh: or keep the vocab of pretrained model -> better result on hmhm
    # asr_model.change_vocabulary(new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type=cfg.model.tokenizer.type)
    # print(asr_model.decoder.vocabulary)

    # bh: reload pretrained model's weight
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    # debug
    # print(next(asr_model.named_parameters()))
    # quit()

    # bh: reload the entire pretrained decoder and joint
    # bh: fastest convergence
    # Insert preserved decoder weights if shapes match
    # if asr_model.decoder.prediction.embed.weight.shape == pretrained_decoder["prediction.embed.weight"].shape:
    #     asr_model.decoder.load_state_dict(pretrained_decoder)
    #     logging.info("Decoder shapes matched - restored weights from pre-trained model")
    # else:
    #     logging.info("\nDecoder shapes did not match - could not restore decoder weights from pre-trained model.")
    # # Insert preserved joint weights if shapes match
    # if asr_model.joint.joint_net[2].weight.shape == pretrained_joint["joint_net.2.weight"].shape:
    #     asr_model.decoder.load_state_dict(pretrained_decoder)
    #     logging.info("Joint shapes matched - restored weights from pre-trained model")
    # else:
    #     logging.info("\nJoint shapes did not match - could not restore joint weights from pre-trained model.")

    # bh: reload part of the pretrained decoder and joint
    # include = [""]
    # exclude = ["decoder.prediction.embed.weight", "joint.joint_net.2.weight", "joint.joint_net.2.bias"]
    # # create dict
    # dict_to_load = {}
    # for k, v in pretrained_decoder.items():
    #     should_add = False
    #     # if any string in include is present, should add
    #     for p in include:
    #         if p in k:
    #             should_add = True
    #             break
    #     # except for if any string from exclude is present
    #     for e in exclude:
    #         if e in k:
    #             # excluded_param_names.append(k)
    #             should_add = False
    #             break
    #     if should_add:
    #         dict_to_load[k] = v
    # todo: add joint
    # asr_model.decoder.load_state_dict(dict_to_load, strict=False)
    # print(asr_model)

    # if freeze encoder
    freeze_encoder = False

    def enable_bn_se(m):
        if type(m) == nn.BatchNorm1d:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

        if "SqueezeExcite" in type(m).__name__:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

    if freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")

    # bh: update the model's config
    # bh: set up new data loaders, optimizers, schedulers, and data augmentation
    # bh: need to be initialized before setup_training_data
    asr_model.cfg = cfg.model

    # setup training, validation, test dataloader after updating tokenizer
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    asr_model.setup_multiple_test_data(cfg.model.test_ds)

    # automatically called by PyTorch Lightning when beginning training the model
    # asr_model.setup_optimization(optim_config=cfg.model.optim)

    # bh: setup augmentation
    asr_model.spec_augmentation = asr_model.from_config_dict(cfg.model.spec_augment)
    # print(asr_model)

    # setup metrics
    # print(asr_model._wer)

    # Setup model with the trainer
    trainer = pl.Trainer(**cfg.trainer)
    asr_model.set_trainer(trainer)

    # ? Finally, update the model's internal config
    # asr_model.cfg = cfg.model

    # Setup Experiment Manager
    exp_manager(trainer, cfg.get("exp_manager", None))

    # train!
    trainer.fit(asr_model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
