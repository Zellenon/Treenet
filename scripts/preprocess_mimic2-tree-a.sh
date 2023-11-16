source defaults.sh
# DATASET=AmazonCat-13K
#DATASET=Wiki-500K
DATASET=MIMIC2

rm $DATA_PATH/$DATASET/vocab*
rm $DATA_PATH/$DATASET/emb_init9-tree.npy
rm $DATA_PATH/$DATASET/emb_init10-tree.npy

python preprocess.py \
--text-path $DATA_PATH/$DATASET/train_icd9_text.txt \
--label-path $DATA_PATH/$DATASET/train_icd9_tree_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab9.npy \
--emb-path $DATA_PATH/$DATASET/emb_init9-tree.npy \
--w2v-model $DATA_PATH/glove.840B.300d.gensim \
--max-len 1000 \
--vocab-size 900000

python preprocess.py \
--text-path $DATA_PATH/$DATASET/train_icd10_text.txt \
--label-path $DATA_PATH/$DATASET/train_icd10_tree_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab10.npy \
--emb-path $DATA_PATH/$DATASET/emb_init10-tree.npy \
--w2v-model $DATA_PATH/glove.840B.300d.gensim \
--max-len 1000 \
--vocab-size 900000

python preprocess.py \
--text-path $DATA_PATH/$DATASET/test_icd9_text.txt \
--label-path $DATA_PATH/$DATASET/test_icd9_tree_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab9.npy \
--max-len 1000 \
--vocab-size 900000

python preprocess.py \
--text-path $DATA_PATH/$DATASET/test_icd10_text.txt \
--label-path $DATA_PATH/$DATASET/test_icd10_tree_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab10.npy \
--max-len 1000 \
--vocab-size 900000
