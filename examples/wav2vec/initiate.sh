cd /home/ubuntu/
mkdir dl4S
cd dl4S
echo Arborescence 
mkdir libs
mkdir data
mkdir model
mkdir results
echo Environnement conda
conda remove --name aws_neuron_mxnet_p36 --all
conda remove --name aws_neuron_pytorch_p36 --all
conda remove --name aws_neuron_tensorflow_p36 --all
conda remove --name pytorch_latest_p37 --all
conda remove --name pytorch_p36 --all
conda create --name p3 --clone python3
source activate p3 
echo Installation des libs 
sudo apt install zip
pip intall torch
cd libs
git clone https://github.com/YamSok/fairseq
cd fairseq
pip install --editable ./
pip install soundfile
pip install editdistance
echo Téléchargement des data
cd ../../data
aws s3 cp s3://dl4s-datasets/WP1.zip ./
aws s3 cp s3://dl4s-datasets/dataset_FR_gen.csv ./
unzip -q WP1.zip
cd ..
echo Génération des datasets de travail
python libs/fairseq/examples/wav2vec/gen_tracker.py 
python libs/fairseq/examples/wav2vec/gen_datasets.py --data data --fairseq libs/fairseq 
cd model
## A ajuster en fonction
aws s3 cp s3://dl4s-datasets/wav2vec_small.pt ./
echo Done !

