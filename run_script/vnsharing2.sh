#!/usr/bin/env bash
# #!/usr/bin/env bash

# vnsharing2ttpro1995@gmail.com
# floyd run --gpu --env pytorch:py2 --data bEd57SbkUir3V7CYC5gjbM:glove --data xsU8MAbJ9TSkpFMMUQ2DLo:sst "sh run_script/vnsharing2.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
# python sentiment.py --name adatree3_2_f2 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 40 --wd 1e-4  --model_name constituency
python sentiment.py --name adatree3_2_f9 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 60 --wd 1e-4  --model_name constituency &
python sentiment.py --name adatree3_2_f10 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 60 --wd 1e-4  --model_name constituency &
wait
echo 'done all -------------'