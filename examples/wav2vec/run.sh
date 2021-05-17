echo Running training on `pwd`/data/WP1_$1 with base_$2 config
fairseq-hydra-train \
    distributed_training.distributed_world_size=1 \
    task.data=`pwd`/data/WP1_$1 \
    model.w2v_path=`pwd`/model/wav2vec_small.pt \
    checkpoint.save_dir=`pwd`/results/ \
    hydra.run.dir=`pwd`/results/ \
    --config-dir `pwd`/libs/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_$2 