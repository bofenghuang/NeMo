#!/usr/bin/env python
# coding=utf-8
# Copyright 2022  Bofeng Huang

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch.nn as nn

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import nemo.collections.asr as nemo_asr


# @hydra_runner(config_path="conf/citrinet/", config_name="config_bpe")
@hydra_runner(config_path="conf/conformer/", config_name="conformer_ctc_bpe_zaion")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    # trainer = pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    # asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # # Initialize the weights of the model from another model, if provided via config
    # asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_fr_conformer_ctc_large")
    # bh: need to specify model class, or can't be loaded
    asr_model = EncDecCTCModelBPE.from_pretrained("stt_fr_conformer_ctc_large", map_location="cpu")
    print(asr_model.summarize())
    # Preserve the decoder parameters in case weight matching can be done later
    pretrained_decoder = asr_model.decoder.state_dict()

    # Update the vocabulary
    asr_model.change_vocabulary(new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type=cfg.model.tokenizer.type)
    # print(asr_model.decoder.vocabulary)

    # Insert preserved model weights if shapes match
    if asr_model.decoder.decoder_layers[0].weight.shape == pretrained_decoder['decoder_layers.0.weight'].shape:
        asr_model.decoder.load_state_dict(pretrained_decoder)
        logging.info("Decoder shapes matched - restored weights from pre-trained model")
    else:
        logging.info("\nDecoder shapes did not match - could not restore decoder weights from pre-trained model.")

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

    # setup training, validation, test dataloader after updating tokenizer
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    asr_model.setup_multiple_test_data(cfg.model.test_ds)

    # automatically called by PyTorch Lightning when beginning training the model
    # asr_model.setup_optimization(optim_config=cfg.model.optim)

    # setup augmentation
    asr_model.spec_augmentation = asr_model.from_config_dict(cfg.model.spec_augment)

    # setup metrics

    # Setup model with the trainer
    trainer = pl.Trainer(**cfg.trainer)
    asr_model.set_trainer(trainer)

    # finally, update the model's internal config
    asr_model.cfg = cfg.model

    # Setup Experiment Manager
    exp_manager(trainer, cfg.get("exp_manager", None))

    # train!
    trainer.fit(asr_model)

    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
