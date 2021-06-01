import os
import sys
sys.path.append("/home/ubuntu/dl4s/libs/py-ctc-decode/")
import ctcdecode
import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
)
import time
import torch
import re
import sys
import argparse
import random
import pandas as pd
import numpy as np


def map_to_array(batch):
    print("Prepare batch")
    speech, _ = torchaudio.load(batch["path"])
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace("’", "'")
    return batch

# def map_to_array(batch):
#     speech, _ = torchaudio.load(batch["file"])
#     batch["speech"] = speech
#     return batch

def map_to_pred(batch):
    print("Process batch")
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    # features = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")

    input_values = features.input_values.to(device)

    attention_mask = features.attention_mask.to(device)
    print("> Model prediction")

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["target"] = batch["sentence"]
    print("> LM")
    text = decoder.decode_batch(logits.cpu())
    batch["corrected"] = text
    # batch["target"] = batch["text"]
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
    ds = load_dataset("common_voice", "fr", split="test[:1%]", data_dir="./cv-corpus-6.1-2020-12-11")
    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    ds = ds.map(map_to_array)
    start = time.time()
    result = ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=list(ds.features.keys()))
    end  = time.time()
    duration = end - start
    # print(result)
    # results = result["train"]
    results = result
    wer_metric = load_metric("wer")
    wer = wer_metric.compute(predictions=results["predicted"], references=results["target"])
    wer_corrected = wer_metric.compute(predictions=results["corrected"], references=results["target"])
    print(f"Dataset : Common-voice FR train split" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer) +
        "Test WER corrected: {:.3f}".format(wer_corrected))
    print("\n")
    show_random_elements(results, out, num_examples=10)
    wer_log = os.path.join(out, "wer_cv.txt")
    with open(wer_log, "w") as err_file:
        print(f"Dataset : Common-voice FR train split" + 
        f"\nInference time : {duration:.2f} \n" + 
        "Test WER: {:.3f}".format(wer), file=err_file +
        "Test WER corrected: {:.3f}".format(wer_corrected))




# processor = Wav2Vec2Processor.from_pretrained(processor_dir)
# model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda")
if __name__ == "__main__":
    print("allo ?")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, type=str,
                        required=True, help="Output dir for wer_log.txt")
    parser.add_argument("--processor", default=None, type=str,
                        required=False, help="Valid ata subset label")
    parser.add_argument("--model", default=None, type=str,
                        required=False, help="Model folder")
    parser.add_argument("--lm", default=None, type=str,
                        required=False, help="Path to kenlm lm")
    args = parser.parse_args()
    processor_dir = args.processor
    model_dir = args.model
    out = args.out
    lm = args.lm

    device = "cuda"
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'  # noqa: W605
    resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french").to(device)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")


    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]
    vocab = [x.lower() for x in vocab]

    vocabulary = vocab
    alpha = 2.5 # LM Weight
    beta = 0.0 # LM Usage Reward
    word_lm_scorer = ctcdecode.WordKenLMScorer(lm, alpha, beta) # use your own kenlm model
    decoder = ctcdecode.BeamSearchDecoder(
        vocabulary,
        num_workers=2,
        beam_width=128,
        scorers=[word_lm_scorer],
        cutoff_prob=np.log(0.000001),
        cutoff_top_n=40
    )

    torch.multiprocessing.set_start_method('spawn')

    main()