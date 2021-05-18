import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import datetime as d

def populate_dict(epoch):
    data = []
    for e in epoch:
        try:
            table = e.split('{')[1][:-1].split(',')
        except:
            print(e)
            input()
        k = [x.split(":")[0].split('"')[1] for x in table]
        v = [float(table[0].split(":")[1])] + [float(x.split(":")[1].split('"')[1]) for x in table if '"' in x.split(":")[1]]
        d = {k:v for (k,v) in zip(k,v)}
        data.append(d)
    return data

def update_df(c):
    # Train
    epoch = []
    for i in range(len(c)):
        if '"epoch"' in c[i] and 'dev' not in c[i]:
            epoch.append(c[i])

    # Validation
    valid_epoch = []
    for i in range(len(c)):
        if 'validation' in c[i] :
            valid_epoch.append(c[i+1])

    data = populate_dict(epoch)
    data = data[2:]
    data = pd.DataFrame.from_dict(data)
    data = data.set_index("epoch")
    
    if valid_epoch:
        valid_data = populate_dict(valid_epoch)
        valid_data = pd.DataFrame.from_dict(valid_data)
        valid_data = valid_data.set_index("epoch")
    else :
        valid_data = []
    return valid_data, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, type=str,
                        required=True, help="Path to hydra_train.log")
    parser.add_argument("--output", default=None, type=str,
                        required=True, help="output dir")
    args = parser.parse_args()
    
    log = args.log
    out = args.output
    
    with open(log, 'r') as inputfile:
        content = inputfile.read()
        
    c = content.split('\n')[258:]
    valid_data, data = update_df(c)
    
    if list(valid_data):
        valid_data.plot(subplots=True, figsize=(20,40))
        timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
        # plt.savefig(os.path.join(out, f"valid_data_{timestamp}.png"))
        plt.savefig(os.path.join(out, f"valid_data.png"))

        plt.close()
    
    data["train_loss"].plot(subplots=True, figsize=(20,10))
    # plt.savefig(os.path.join(out, f"train_loss_{timestamp}.png"))
    plt.savefig(os.path.join(out, f"train_loss.png"))
    plt.close()
    
    data.plot(subplots=True, figsize=(20,40))    
    # plt.savefig(os.path.join(out, f"train_data_{timestamp}.png"))
    plt.savefig(os.path.join(out, f"train_data.png"))
    plt.close()
    
if __name__ == "__main__":
    main()
    