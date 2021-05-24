import os
import json
import argparse
import torch
import torchaudio
import librosa
import pandas as pd
import glob
import numpy as np
import IPython.display as ipd
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import soundfile as sf
from jiwer import wer
# import torchaudio

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, \
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, load_metric

def format_csv_tracker(source, source_path, output):
    df = pd.read_csv(source)
    df["file"] = df["file"].apply(lambda x : os.path.join(source_path, x))
    df["text"] = df["transcription"]
    # df["text"] = df["transcription"].apply(lambda x : x.upper())
    df = df.drop("transcription", axis=1)
    df.to_csv(output, index=False)


def import_data():
    
    format_csv_tracker(TRAIN_CSV_RAW, TRAIN_PATH, TRAIN_CSV)
    format_csv_tracker(VALID_CSV_RAW, VALID_PATH, VALID_CSV)

    data = load_dataset('csv', data_files={'train': TRAIN_CSV,'test': VALID_CSV})
    print(data)
    return data

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def gen_vocab(data):
    vocabs = data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, \
        remove_columns=data.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open(f'results_hg/{MODEL}/{LABEL}/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

def speech_file_to_array_fn(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    speech_array, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    # batch["target_text"] = batch["text"]
    batch["target_text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower().replace("’", "'")

    return batch

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], \
        sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

def compute_metrics(pred):
    wer_metric = load_metric("wer")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def data_preparation():
    data = import_data()
    global processor

    if glob.glob(f"results_hg/{MODEL}/{LABEL}/processor/*"):
        print(">> From pretrained processor ")
        processor = Wav2Vec2Processor.from_pretrained(f"results_hg/{MODEL}/{LABEL}/processor")
    else :
        print(">> Creating processor ")

        gen_vocab(data)
        tokenizer = Wav2Vec2CTCTokenizer(f"results_hg/{MODEL}/{LABEL}/vocab.json", unk_token="[UNK]", \
            pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, \
            sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(f'results_hg/{MODEL}/{LABEL}/processor/')

    dataset = data.map(speech_file_to_array_fn, \
         remove_columns=data.column_names["train"], num_proc=4)
    dataset_prepared = dataset.map(prepare_dataset, \
        remove_columns=dataset.column_names["train"], batch_size=8, num_proc=4, batched=True)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    return processor, dataset_prepared, data_collator

def remove_special_characters(batch):
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

def speech_file_to_array_fn_v2(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch

def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch

def data_preparation_v2():
    common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
    common_voice_test = load_dataset("common_voice", "tr", split="test")
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_train = common_voice_train.map(remove_special_characters)
    common_voice_test = common_voice_test.map(remove_special_characters)

    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(f'results_hg/{MODEL}/{LABEL}/vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    global processor

    print(">> Creating processor ")

    tokenizer = Wav2Vec2CTCTokenizer(f"results_hg/{MODEL}/{LABEL}/vocab.json", unk_token="[UNK]", \
        pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, \
        sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(f'results_hg/{MODEL}/{LABEL}/processor/')

    common_voice_train = common_voice_train.map(speech_file_to_array_fn_v2, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(speech_file_to_array_fn_v2, remove_columns=common_voice_test.column_names)
    
    common_voice_train = common_voice_train.map(resample, num_proc=4)
    common_voice_test = common_voice_test.map(resample, num_proc=4)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)

    return processor, common_voice_train, common_voice_test, data_collator

def main(checkpoint):

    processor, dataset_prepared, data_collator = data_preparation()
    # processor, common_voice_train, common_voice_test, data_collator = data_preparation_v2()


    model_str = "facebook/wav2vec2-base" if MODEL == "base" else "facebook/wav2vec2-large-xlsr-53"

    print(">> Starting fine-tuning on model " + model_str )
    print(">> Training dataset :", LABEL)
    print("\n\n")

    model = Wav2Vec2ForCTC.from_pretrained(
        model_str, 
        # attention_dropout=0.1,
        # hidden_dropout=0.1,
        # feat_proj_dropout=0.0,
        # mask_time_prob=0.05,
        # layerdrop=0.1,
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )

    model.freeze_feature_extractor()
    
    training_args = TrainingArguments(
        output_dir=f"/home/ubuntu/dl4s/results_hg/{MODEL}/{LABEL}/",
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=300,
        fp16=True,
        save_strategy = "steps",
        save_steps=400,
        eval_steps=400,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=3e-4,
        warmup_steps=250,
        save_total_limit=2,
    )

    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_prepared['train'],
    eval_dataset=dataset_prepared['test'],
    tokenizer=processor.feature_extractor,
    )

    if checkpoint :
        print(">> Resume from checkpoint", checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
    else :
        print(">> Initiate fine-tuning")
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, type=str,
                        required=True, help="Train data tracker csv")
    parser.add_argument("--valid", default=None, type=str,
                        required=True, help="Valid data tracker csv")
    parser.add_argument("--model", default=None, type=str,
                        required=True, help="Pretrained model (base / xlsr)")
    parser.add_argument("--checkpoint", default=None, type=str,
                        required=False, help="Pretrained model (base / xlsr)")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    TRAIN_CSV_RAW = args.train
    VALID_CSV_RAW = args.valid  
    MODEL = args.model #Facultatif : sert à ranger les modèles dans les bons dossiers

    TRAIN_PATH = TRAIN_CSV_RAW.split("dataset")[0]
    VALID_PATH = VALID_CSV_RAW.split("dataset")[0]

    TRAIN_CSV = os.path.join(TRAIN_PATH, "train_hg.csv")
    VALID_CSV = os.path.join(VALID_PATH, "valid_hg.csv")

    LABEL = TRAIN_PATH.split('_')[-1][:-1]
    # LABEL = "commonvoice"

########
########
# Attention : modifie le chemin  f"results_hg/{MODEL}/{LABEL}/
# qui apparaît plusieurs fois : c'est là que tout s'enregistre 
# processor, checkpints, vocab ...

# Attention aussi à mettre à jour les tracker csv dataset_FR_[label].csv

# Ex usage du script : python hg_trainer --train data/WP1_15m/dataset_FR_15m.csv --valid data/WP1_valid_small/dataset_FR_valid_small.csv

    main(checkpoint)
