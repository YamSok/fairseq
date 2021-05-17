source activate p3

while true;
do
    sleep $2
    python libs/fairseq/examples/wav2vec/training_monitor.py --log results/hydra_train.log --output results/
    zip graphs *.png
    aws s3 cp results/hydra_train.log s3://dl4s-datasets/0$1-backup/
    aws s3 cp results/graphs.zip s3://dl4s-datasets/0$1-backup/
done