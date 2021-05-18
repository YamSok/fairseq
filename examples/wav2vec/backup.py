from subprocess import run
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dest", default=None, type=str,
                    required=True, help="S3 bucket to save to")
parser.add_argument("--sleep", default=None, type=str,
                    required=True, help="time (s) between each backup")
args = parser.parse_args()
dest = args.dest
sleep_time = args.sleep

run('touch results/placeholder.png', shell=True)

while True:
    run("python libs/fairseq/examples/wav2vec/training_monitor.py --log results/hydra_train.log --output results/", shell=True)
    run("zip results/graphs results/*.png", shell=True)
    run(f"aws s3 cp results/checkpoint_last.pt s3://dl4s-datasets/0{dest}-backup/", shell=True)
    run(f"aws s3 cp results/hydra_train.log s3://dl4s-datasets/0{dest}-backup/", shell=True)
    run(f"aws s3 cp results/graphs.zip s3://dl4s-datasets/0{dest}-backup/", shell=True)
    time.sleep(int(sleep_time))
