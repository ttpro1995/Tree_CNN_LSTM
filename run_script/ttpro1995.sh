# #!/usr/bin/env bash

# ttpro1995
# floyd run --gpu --env pytorch:py2 --data Nw6RTRWwtZiSZknL6tUsjc:glove --data P9SWxDxfKMxvFTRhTwbcKB:sst "sh run_script/ttpro1995.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name constituency
echo "done gen dataset"

python sentiment.py --name adatree3_2_f4 --data sst/ --glove glove/ --logs /output  --saved /output  --optim adagrad --lr 0.01 --emblr 0.1 --epochs 60 --wd 1e-4  --model_name constituency
echo 'done all -------------'