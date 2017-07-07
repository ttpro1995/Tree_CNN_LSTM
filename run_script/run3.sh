# #!/usr/bin/env bash

# upload1ttpro1995@gmail.com
# floyd run --gpu --env pytorch:py2 --data DzZLqsw28Hr3FGdGT4GNnc:glove --data AV5ntSwiV5gikmVxS3pKhS:sst "sh run_script/run3.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name constituency
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
# python sentiment.py --name adatree3_f  --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 40 --wd 1e-4  --model_name constituency --epochs 50
python sentiment.py --name adatree6_f  --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.01 --epochs 50 --wd 1e-4  --model_name constituency


echo 'done all -------------'