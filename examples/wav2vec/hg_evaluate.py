import soundfile as sf
import torch
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset, Dataset, load_metric
import pandas as pd
import os
from jiwer import wer

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

def import_data(source, source_path, output):
    df = pd.read_csv(source)
    df["file"] = df["file"].apply(lambda x : os.path.join(source_path, x))
    df["text"] = df["transcription"]
    # df["text"] = df["transcription"].apply(lambda x : x.upper())
    df = df.drop("transcription", axis=1)
    df.to_csv(output, index=False)
    data = load_dataset('csv', data_files=output)
    data = data.map(map_to_array)
    return data

def map_to_result(batch):
    
    if torch.cuda.is_available():
        model.to("cuda")
        input_values = processor(
            batch["speech"], 
            sampling_rate=batch["sampling_rate"], 
            return_tensors="pt"
        ).input_values.to("cuda")
    else:
        input_values = processor(
            batch["speech"], 
            sampling_rate=batch["sampling_rate"], 
            return_tensors="pt"
        ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  
    return batch


def main(out):
    data = import_data(TEST_CSV_RAW, TEST_PATH, TEST_CSV)

    results = data.map(map_to_result)
    wer_metric = load_metric("wer")

    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["transcription"])))
    wer_log = os.path.join(out, "wer.txt")
    with open(wer_log, "w") as err_file:
        print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["transcription"])), file=err_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None, type=str,
                        required=True, help="Train data tracker csv")
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

    # processor = Wav2Vec2Processor.from_pretrained(processor_dir)
    # model = Wav2Vec2ForCTC.from_pretrained(model_dir)

    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

    # ## add base et xlsr base
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

    TEST_CSV_RAW = args.test
    TEST_PATH = TEST_CSV_RAW.split("dataset")[0]
    TEST_CSV = os.path.join(TEST_PATH, "test_hg.csv")

    main(out)