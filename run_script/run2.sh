# #!/usr/bin/env bash

# ttpro1995
# floyd run --gpu --env pytorch:py2 --data Nw6RTRWwtZiSZknL6tUsjc:glove --data P9SWxDxfKMxvFTRhTwbcKB:sst "sh run_script/run2.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name constituency
echo "done gen dataset"

python sentiment.py --name adagrad5 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0 --epochs 30 --wd 0 --model_name constituency &
python sentiment.py --name adagrad6 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0 --epochs 30 --wd 1e-4 --model_name constituency &
python sentiment.py --name adam2 --data sst/ --glove glove/ --logs /output  --saved /output --optim adam --lr 0.001 --emblr 0.1 --epochs 30 --wd 0 --model_name constituency &
python sentiment.py --name adam3 --data sst/ --glove glove/ --logs /output  --saved /output --optim adam --lr 0.001 --emblr 0 --epochs 30 --wd 0 --model_name constituency &
python sentiment.py --name adam4 --data sst/ --glove glove/ --logs /output  --saved /output --optim adam --lr 0.001 --emblr 0.1 --epochs 30 --wd 1e-6 --model_name constituency &
python sentiment.py --name adam5 --data sst/ --glove glove/ --logs /output  --saved /output --optim adam --lr 0.001 --emblr 0 --epochs 30 --wd 1e-6 --model_name constituency &
wait

echo 'done all -------------'