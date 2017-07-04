# #!/usr/bin/env bash

# ttpro1995
# floyd run --gpu --env pytorch:py2 --data Nw6RTRWwtZiSZknL6tUsjc:glove --data P9SWxDxfKMxvFTRhTwbcKB:sst "sh run_script/run2.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name constituency
echo "done gen dataset"

python sentiment.py --name adagradlstm8 --data sst/ --glove glove/ --logs /output  --saved /output --optim adagrad --lr 0.05 --emblr 0.05 --epochs 20 --wd 1e-4 --train_subtrees -1 --model_name lstm

echo 'done all -------------'