# #!/usr/bin/env bash

# upload1ttpro1995@gmail.com
# floyd run --gpu --env pytorch:py2 --data f9rfYfFCmoUmZX8AW6CdRL:glove --data ArDQxv9EDjDi2ub4waudWJ:sst "sh run_script/uploadttpro1995.sh"

pip install -U meowlogtool
cp -r /sst .
cp -r /glove .

python sentiment.py --name gen_ds --data sst/ --glove glove/ --logs /output   --model_name lstm
echo "done gen dataset"

# python sentiment.py --name adadeltalstm1 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --wd 0 --epochs 60 --model_name lstm
python sentiment.py --name adadelta2 --data sst/ --glove glove/ --logs /output  --saved /output --optim adadelta --lr 1 --emblr 0 --epochs 60 --wd 1e-4 --model_name lstm

echo 'done all -------------'