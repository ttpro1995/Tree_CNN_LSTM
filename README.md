###

Implement cnn layer to tree-lstm/lstm

Install dependencies:
```
pip install meowlogtool
pip install tqdm
conda install pytorch torchvision cuda80 -c soumith
```


```
--data preprocessed sst folder
--name id of experiment, would become log file
--lr  model learning rate
--emblr embedding layer learning rate
--epochs number of epochs
--model_name lstm/bilstm/constituency
--wd weight decay (L2 regularization)
--embedding glove/multi_channel
--glove folder of glove (in case embedding is glove)
--embedding_other in case embedding is multi_channel, embedding of first channel (glove format)
--embedding_othert in case embedding is multi_channel, embedding of second channel (glove format)
```


```
python sentiment.py --data data/sst_seq --name lstm3_2c_nope --optim adagrad --lr 0.01 --emblr 0.1 --epochs 21 --wd 1e-4 --model_name lstm --embedding multi_channel --embedding_othert /media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted/glove.840B.300d --embedding_other ../treelstm.pytorch/data/glove/glove.840B.300d
```

### License
MIT