
```bash
# prep nemo manifest and upsample & save audio segments
./prep_csv_to_manifest.py \
    --input_file_path /home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv \
    --manifest_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
    --out_wav_dir /home/bhuang/corpus/speech/internal/hm_hm_16k/audios_16k/train_hmhm_merged_and_raw

# normalize text column in manifest
python prep_normalize_text.py \
    --in_file_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
    --out_file_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw_new.json

# tokenize cumstomized dataset
# ?? or use --spe_type="bpe" \
python [NEMO_GIT_FOLDER]/scripts/tokenizers/process_asr_text_tokenizer.py \
 --manifest="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm.json" \
 --data_root="/home/bhuang/asr/NeMo/examples/asr/outputs/tokenizer_unigram128_hmhm" \
 --vocab_size=128 \
 --tokenizer="spe" \
 --spe_type="unigram" \
 --spe_character_coverage=1.0 \
 --no_lower_case \
 --log


# 
```

unigram vocab on hm-hm is weird
