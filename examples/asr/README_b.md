

## Script Usage

*Prep nemo manifest and upsample & save audio segments (Zaion data)*

```bash
scripts/prep_csv_to_manifest.py \
    --input_file_path /home/bhuang/corpus/speech/internal/hm_hm_merged/train_hmhm_merged_and_raw.csv \
    --manifest_path /home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm_merged_and_raw.json \
    --out_wav_dir /home/bhuang/corpus/speech/internal/hm_hm_16k/audios_16k/train_hmhm_merged_and_raw
```
*Tokenize cumstomized dataset*

```bash
# ?? or use --spe_type="bpe" \
python scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="/home/bhuang/corpus/speech/internal/hm_hm_16k/manifest_nemo/train_hmhm.json" \
    --data_root="/home/bhuang/asr/NeMo/examples/asr/outputs/tokenizer_unigram128_hmhm" \
    --vocab_size=128 \
    --tokenizer="spe" \
    --spe_type="unigram" \
    --spe_character_coverage=1.0 \
    --no_lower_case \
    --log
```

unigram vocab on hm-hm is weird

## Train from Scratch

Data Collection and Curation

```bash
# download and convert dataset to NeMo's manifest file
scripts/convert_hf_dataset_to_nemo.sh

# normalize text and eda
# normalize early to debug easily
scripts/normalize_dataset.sh

# dedup special datasets
python scripts/preprocess_dataset.py /projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized.json /projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_min05_dedup4.json --min_duration_s 0.5 --max_identical_text 4
python scripts/preprocess_dataset.py /projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized.json /projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized_min1_dedup256.json --min_duration_s 1 --max_identical_text 256

# merge into one manifest
python scripts/merge_datasets.py

# preprocess data
python scripts/preprocess_dataset.py /projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr.json /projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256.json --max_identical_text 256

```

<!-- Number of Speakers, Min. Duration, Max. Duration -->

|            Dataset            | Number of Files | Total Duration | Avg. Duration | Punctuation | Casing | Description                                                     |
| :---------------------------: | :-------------: | :------------: | :-----------: | :---------: | :----: | --------------------------------------------------------------- |
|        MCV-13/fr/train        |     509,300     |    732.02h     |     5.17s     |      ✅      |   ✅    | Crowd workers recording text from Wikipedia                     |
|     MCV-13/fr/validation      |     16,114      |     25.81h     |     5.77s     |      ✅      |   ✅    |                                                                 |
|        MCV-13/fr/test         |     16,114      |     26.21h     |     5.86s     |      ✅      |   ✅    |                                                                 |
|         MLS/fr/train          |     258,213     |    1076.58h    |    15.01s     |      ❌      |   ❌    | LibriVox read audiobooks                                        |
|       MLS/fr/validation       |      2,416      |     10.07h     |    15.01s     |      ❌      |   ❌    |                                                                 |
|          MLS/fr/test          |      2,426      |     10.07h     |    14.94s     |      ❌      |   ❌    |                                                                 |
|      Voxpopuli/fr/train       |     73,561      |    205.70h     |    10.07s     |      ✅      |   ❌?   | European Parliament event recordings (2009-2020)                |
|    Voxpopuli/fr/validation    |      1,727      |     4.96h      |    10.35s     |      ✅      |   ❌?   |                                                                 |
|       Voxpopuli/fr/test       |      1,742      |     4.89h      |    10.12s     |      ✅      |   ❌?   |                                                                 |
|        Fleurs/fr/train        |      3,193      |     10.32h     |    11.64s     |      ✅      |   ❌    | FLoRes in 102 languages                                         |
|     Fleurs/fr/validation      |       289       |     0.80h      |     9.91s     |      ✅      |   ❌    |                                                                 |
|        Fleurs/fr/test         |       676       |     1.95h      |    10.39s     |      ✅      |   ❌    |                                                                 |
|        mTEDx/fr/train         |     116,045     |    175.83h     |     5.45s     |      ✅      |   ✅    | TEDx talks                                                      |
|      mTEDx/fr/validation      |      1,036      |     1.81h      |     6.28s     |      ✅      |   ✅    |                                                                 |
|         mTEDx/fr/test         |      1,059      |     1.55h      |     5.28s     |      ✅      |   ✅    |                                                                 |
|        MediaSpeech/fr         |      2,498      |     10.00h     |    14.41s     |      ❌      |   ❌    | short speech segments extracted from YouTube                    |
|          M-AILABS/fr          |     86,597      |    181.90h     |     7.56s     |      ✅      |   ✅    | Most of the data is based on LibriVox and Project Gutenberg     |
| African-Accented-French/train |      9,401      |     11.68h     |     4.47s     |      ✅      |   ❌    | From Cameroon, Chad, Congo, Gabon, and Niger                    |
| African-Accented-French/test  |      1,985      |     1.69h      |     3.07s     |      ✅      |   ❌    |                                                                 |
|        Lingua-Libre/fr        |     257,927     |     91.52h     |     1.28s     |      ❌      |   ❌    | Wikimédia France, short audios                                  |
|           Att-HACK            |     36,634      |     27.12h     |     2.66s     |      ❌      |   ❌    | Acted expressive speech in French (from 3 to 5 for each phrase) |
|        PolyAI/minds14         |       539       |     1.25h      |     8.36s     |      ❌      |   ❌    | SLU in e-banking domain                                         |
|             Total             |    1,353,908    |    2523.92h    |     6.71s     |             |        |                                                                 |


# todo: empty, dedup, short
# repeated in Att-HACK, Lingua-Libre
# mailabs overlap

```bash
# estimate tokenizer
python ../../scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256.json" \
    --data_root="nemo_experiments/tokenizers_general" \
    --vocab_size=1024 \
    --tokenizer="spe" \
    --spe_type="unigram" \
    --spe_character_coverage=1.0 \
    --log

# bucket datasets
# may result into training speeedup of more than 2X
python ../../scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
    --manifest_path="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256.json" \
    --target_dir="/projects/bhuang/corpus/speech/nemo_manifests/final/2023-09-14/train_asr_processed_dedup256_tarred" \
    --num_shards=8 \
    --buckets_num=4 \
    --max_duration=30.0 \
    --min_duration=0.1 \
    --shuffle --shuffle_seed=1 \
    --sort_in_shards \
    --workers=8
```