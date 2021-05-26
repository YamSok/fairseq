import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import time
import torch
import re
import sys
import argparse
import random
import pandas as pd
import os
def map_to_array(batch):
    speech, _ = torchaudio.load(batch["path"])
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace("’", "'")
    return batch


def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["target"] = batch["sentence"]
    return batch

def show_random_elements(dataset, out, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
        
    df = pd.DataFrame(dataset[picks])[['target', 'predicted']]
    print(df)
    example_log = os.path.join(out, "example_cv.txt")
    with open(example_log, "w") as ex_log:
        print(df, file=ex_log)

def main():
    ds = load_dataset("common_voice", "fr", split="test[:40%]", data_dir="./cv-corpus-6.1-2020-12-11")
    ds = ds.map(map_to_array)
    start = time.time()
    result = ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=list(ds.features.keys()))
    wer = load_metric("wer")
    end  = time.time()
    duration = end - start
    print(result)
    # results = result["train"]
    results = result
    wer_metric = load_metric("wer")

    print(f"Dataset : Common-voice FR train split" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer_metric.compute(predictions=results["predicted"], references=results["target"])))
    print("\n")
    show_random_elements(results, out, num_examples=10)
    wer_log = os.path.join(out, "wer_cv.txt")
    with open(wer_log, "w") as err_file:
        print(f"Dataset : Common-voice FR train split" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer_metric.compute(predictions=results["predicted"], references=results["target"])), file=err_file)

    print(wer.compute(predictions=result["predicted"], references=result["target"]))

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None, type=str,
                    required=True, help="Output dir for wer_log.txt")
parser.add_argument("--processor", default=None, type=str,
                    required=False, help="Valid ata subset label")
parser.add_argument("--model", default=None, type=str,
                    required=False, help="Model folder")
                 
args = parser.parse_args()
processor_dir = args.processor
model_dir = args.model
out = args.out

device = "cuda"
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'  # noqa: W605
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
processor = Wav2Vec2Processor.from_pretrained(processor_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda")

main()