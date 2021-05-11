import pandas as pd
import glob 
from subprocess import run


def delete_untracked(data):
    files_absolute = glob.glob("data/WP1/*.wav")
    files_relative = [f.split("/")[-1] for f in files_absolute]
    count = 0
    for i in range(len(files_relative)):
        if files_relative[i] not in data["file"].to_list():
            run(f'rm -f {files_absolute[i]}', shell=True)
            count += 1
    print(count)

def gen_trans(data):
    with open('data/transcript.txt', "w") as trans:
        for index, row in data.iterrows():
            texts = row["file"].split(".wav")[0] + " " + row["transcription"]
            print(texts, file=trans)
    
    
def gen_dict():     
    save_dir = "data/"
    transcript_file = "data/transcript.txt"
    
    dictionary = os.path.join(save_dir,'dict.ltr.txt')
    
    
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

    
def main():
    # Import et nettoyage
    csv_path = 'data/dataset_FR_gen.csv'
    data = pd.read_csv(csv_path)
    data = data.drop(["category", "topic"], axis=1)
    data = data.dropna()

    # Sélection des contributeurs fiables
    # data = data[data["file"].apply(lambda x : "CLEMENT" in x or "MONTA" in x or "GOULARD" in x)]
    data = data[data["file"].apply(lambda x : "MONTA" in x)]


    # Supprime les noms de fichiers avec espace
    # data = data.drop(data[data['file'].str.contains("ARN")].index)

    # On enlève les chevrons qui apparaissent dans les "<euh>"
    transcription = data["transcription"].apply(lambda x : x.replace("<", "").replace(">", ""))
    data["transcription"] = transcription

    # On supprime les fichiers non présents dans le tracker
    delete_untracked(data)

    # Save
    data.to_csv("data/dataset_FR_filtered.csv", index=False)
    
    gen_trans(data)
    gen_dict()
    
main()