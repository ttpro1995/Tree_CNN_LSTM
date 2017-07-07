#!/usr/bin/env bash
# i1187093@mvrht.net

# floyd run --gpu --env pytorch:py2 --data WTbJpZppSeRQ4kHw637B6V:glove --data sx7duzYg8x7P9nnM4GGV8C:sst "sh run_script/i1187093.sh"
pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
# python sentiment.py --name adadelta2 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --epochs 60 --wd 1e-4 --model_name lstm
python sentiment.py --name adatree3_2_f --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 40 --wd 1e-4  --model_name constituency
echo 'done all -------------'