# #!/usr/bin/env bash


# floyd run --gpu --env pytorch:py2 --data DXopjCBnJhPQoxF52zFMDn:glove --data GnAkTUvL3AJ57FaWTzHxYC:sst "sh run_script/fine_tune_floyd.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

#python sentiment.py --name adagrad1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0.1 --wd 1e-4 --epochs 20 --model_name constituency &
#python sentiment.py --name adagrad2 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 20 --wd 1e-4 --model_name constituency &
#python sentiment.py --name adagrad3 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 20 --wd 0 --model_name constituency &
#python sentiment.py --name adagradlstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0.1 --wd 1e-4 --epochs 20 --model_name lstm &
#python sentiment.py --name adagradlstm2 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 20 --wd 1e-4 --model_name lstm &
#python sentiment.py --name adagradlstm3 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 20 --wd 0 --model_name lstm &
python sentiment.py --name adagradlstm6 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0.1 --wd 1e-4 --epochs 20 --train_subtrees -1 --model_name lstm &
python sentiment.py --name adagradlstm7 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.01 --emblr 0.1 --epochs 20 --wd 1e-4 --train_subtrees -1 --model_name lstm &


wait
echo 'done all -------------'