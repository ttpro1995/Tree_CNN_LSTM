#!/usr/bin/env bash
# #!/usr/bin/env bash

# vnsharing4
# floyd run --gpu --env pytorch:py2 --data ahfzDXwWFJANStWEuCZZ99:glove --data nNz6AEzXhheTp9MvyUNQjJ:sst "sh run_script/vnsharing4.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
# python sentiment.py --name adatree3_2_f2 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 40 --wd 1e-4  --model_name constituency
python sentiment.py --name adadou_f1 --optim adagrad --lr 0.01 --emblr 0.01 --epochs 60 --wd 1e-4  --model_name constituency --embedding multi_channel --embedding_other /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted/glove.840B.300d --embedding_othert ../treelstm.pytorch/data/glove/glove.840B.300d &
python sentiment.py --name adadou_f2 --optim adagrad --lr 0.01 --emblr 0.01 --epochs 60 --wd 1e-4  --model_name constituency --embedding multi_channel --embedding_other /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted/glove.840B.300d --embedding_othert ../treelstm.pytorch/data/glove/glove.840B.300d &
wait
echo 'done all -------------'