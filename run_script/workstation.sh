#!/usr/bin/env bash
# #!/usr/bin/env bash

# workstation.hahattpro@gmail.com
# floyd run --gpu --env pytorch:py2 --data XBdbxB4VvjSZAfijpjfFrV:glove --data FywEquWgbfGC8zMChERsi2:sst "sh run_script/workstation.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
#python sentiment.py --name adatree5_2_f --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.01 --epochs 40 --wd 1e-4  --model_name constituency
python sentiment.py --name adatree3_2_f5 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 60 --wd 1e-4  --model_name constituency
echo 'done all -------------'
