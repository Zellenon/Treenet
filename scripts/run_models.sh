#!/usr/bin/env bash

source defaults.sh

#DATASET=AmazonCat-13K
#DATASET=Wiki-500K
# DATASET=MIMIC

#MODEL=XMLCNN
#MODEL=CorNetXMLCNN
#MODEL=BertXML
#MODEL=CorNetBertXML
# MODEL=MeSHProbeNet
#MODEL=AttentionXML
#MODEL=CorNetAttentionXML

# CUDA_LAUNCH_BLOCKING=1 python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml
python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml

python evaluation.py \
--results $DATA_PATH/$DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATA_PATH/$DATASET/test_labels.npy \
--train-labels $DATA_PATH/$DATASET/train_labels.npy
