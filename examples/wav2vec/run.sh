echo Validation `pwd`/data/WP1_$3
cp -v `pwd`/data/WP1_$3/train.tsv `pwd`/data/WP1_$1/dev_other.tsv
cp -v `pwd`/data/WP1_$3/train.ltr `pwd`/data/WP1_$1/dev_other.ltr
cp -v `pwd`/data/WP1_$3/train.wrd `pwd`/data/WP1_$1/dev_other.wrd
echo Running training on `pwd`/data/WP1_$1 with base_$2 config
source activate p3
sleep 4
nohup fairseq-hydra-train \
    distributed_training.distributed_world_size=1 \
    task.data=`pwd`/data/WP1_$1 \
    model.w2v_path=`pwd`/model/wav2vec_small.pt \
    checkpoint.save_dir=`pwd`/results/ \
    hydra.run.dir=`pwd`/results/ \
    --config-dir `pwd`/libs/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_$2 &