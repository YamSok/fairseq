source activate p3

while true;
do
    sleep $2
    echo Generate graphs
    python libs/fairseq/examples/wav2vec/training_monitor.py --log results/hydra_train.log --output results/
    sleep 2
    echo Zip graphs
    zip graphs results/*.png
    echo Upload to S3
    aws s3 cp results/hydra_train.log s3://dl4s-datasets/0$1-backup/
    aws s3 cp results/graphs.zip s3://dl4s-datasets/0$1-backup/
done