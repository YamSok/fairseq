import pandas as pd
import glob 
from subprocess import run
import argparse
from sklearn.model_selection import train_test_split
import os
from collections import Counter


def gen_train_test(data, percent):
    X = data["file"]
    y = data["transcription"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=percent)

    test = pd.DataFrame(columns=["file", "transcription"])
    test["file"] = X_test
    test["transcription"] = y_test
    return train, test

def delete_untracked(data, folder):
    files_absolute = glob.glob(folder + "/*.wav")
    files_relative = [f.split("/")[-1] for f in files_absolute]
    count = 0
    run(f'rm -f {folder}/*GOULARD*', shell=True) # Bug filename avec espace sinon
    for i in range(len(files_relative)):
        if files_relative[i] not in data["file"].to_list():
            run(f'rm -f {files_absolute[i]}', shell=True)
            count += 1
    print(count)
    
def gen_trans(data, folder):
    """
    Generate transcript from tracker csv
    Format : filename transcription
    """
    with open(folder + '/transcript.txt', "w") as trans:
        for index, row in data.iterrows():
            texts = row["file"].split(".wav")[0] + " " + row["transcription"]
            print(texts, file=trans)
    

def gen_dict(folder):     
    """
    Fonction issue dut git de mailong25 (STT vietnamien)
    Generate dict from transcript.txt
    """
    save_dir = folder
    transcript_file = os.path.join(folder, "transcript.txt")
    
    dictionary = os.path.join(save_dir, 'dict.ltr.txt')
    
    with open(transcript_file) as f:
        data = f.read().splitlines()
        
    words = [d.split(' ')[1].upper() for d in data]
    letters = [d.replace(' ','|') for d in words]
    letters = [' '.join(list(d)) + ' |' for d in letters]
    chars = [l.split() for l in letters]
    chars = [j for i in chars for j in i]
    char_stats = list(Counter(chars).items())
    char_stats = sorted(char_stats, key=lambda x : x[1], reverse = True)
    char_stats = [c[0] + ' ' + str(c[1]) for c in char_stats]
    
    with open(dictionary,'w') as f:
        f.write('\n'.join(char_stats))

def populate(data, label, percent, folder, fairseq):
    print({label})
    print("Copie")
    run("cp -r " + os.path.join(folder, f"WP1 {folder}/WP1_{label}"), shell=True)
    train, test = gen_train_test(data, percent)
    print(f"Save tracker {folder}/WP1_{label}/dataset_FR_{label}.csv")
    test.to_csv(os.path.join(folder, f"WP1_{label}/dataset_FR_{label}.csv"), index=False)    
    print("Delete untracked")
    delete_untracked(test, os.path.join(folder, f"WP1_{label}"))
    print("gen_trans")
    gen_trans(test, folder)
    print("gen_dict")
    gen_dict(folder)
    print("Manifest")
    run('python ' + 
        os.path.join(fairseq, "examples/wav2vec/wav2vec_manifest.py") + " " + os.path.join(folder, f"WP1_{label}") +
        " --dest " + os.path.join(folder, f"WP1_{label}") + 
        " --ext wav --valid-percent 0", shell=True)
    
#     print("Rename")
#     run('mv ' os.path.join(folder, f'WP1_{label}/valid.tsv') + ' ' + os.path.join(folder, f'WP1_{label}/dev_other.tsv'), shell=True)
    
    print("train.tsv")
    run('python ' + os.path.join(fairseq, 'examples/wav2vec/dl4s_labels.py') + ' ' + os.path.join(folder, f'WP1_{label}/train.tsv') +
    ' --output-dir ' + os.path.join(folder, f'WP1_{label}') +
    ' --output-name train --tracker ' + os.path.join(folder, f'WP1_{label}/dataset_FR_{label}.csv'), shell=True)
    
    # Partie non nécessaire car on passe par un dataset fixe pour la validation
    
#     print("dev.tsv")
    
#     run(f'{fairseq}/example/wav2vec/libri_labels.py {folder}/WP_{label}/dev_other.tsv \
#     --output-dir {folder}/WP_{label} \
#     --output-name dev_other \
#     --tracker {folder}/WP1_{label}/dataset_FR_{label}.csv ')
        
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str,
                        required=True, help="Path to data")
    parser.add_argument("--fairseq", default=None, type=str,
                        required=True, help="Path to fairseq")
    args = parser.parse_args()
    folder = args.data
    fairseq = args.fairseq
    
    # Import du tracker clean avec juste Charline
    data = pd.read_csv(os.path.join(folder, "dataset_FR_filtered.csv"))
    
    # Fraction calculée par rapport à la taille du dataset de base pour Charline : ~15h
    populate(data, '5h', 1/3, folder, fairseq)
    populate(data, '2h', 1/7, folder, fairseq)
    populate(data, '1h', 1/15, folder, fairseq)
    populate(data, '30', 1/30, folder, fairseq)
    populate(data, '15m', 1/60, folder, fairseq)
    
    populate(data, 'valid', 1/3, folder, fairseq)
    
        
main()