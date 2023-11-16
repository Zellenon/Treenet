source defaults.sh
# DATASET=AmazonCat-13K
#DATASET=Wiki-500K

rm $DATA_PATH/$DATASET/vocab9.npy
rm $DATA_PATH/$DATASET/emb_init9.npy
rm $DATA_PATH/$DATASET/vocab10.npy
rm $DATA_PATH/$DATASET/emb_init10.npy

python preprocess.py \
--text-path $DATA_PATH/$DATASET/train_icd9_text.txt \
--label-path $DATA_PATH/$DATASET/train_icd9_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab9.npy \
--emb-path $DATA_PATH/$DATASET/emb_init9.npy \
--w2v-model $DATA_PATH/glove.840B.300d.gensim

python preprocess.py \
--text-path $DATA_PATH/$DATASET/train_icd10_text.txt \
--label-path $DATA_PATH/$DATASET/train_icd10_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab10.npy \
--emb-path $DATA_PATH/$DATASET/emb_init10.npy \
--w2v-model $DATA_PATH/glove.840B.300d.gensim

python preprocess.py \
--text-path $DATA_PATH/$DATASET/test_icd9_text.txt \
--label-path $DATA_PATH/$DATASET/test_icd9_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab9.npy

python preprocess.py \
--text-path $DATA_PATH/$DATASET/test_icd10_text.txt \
--label-path $DATA_PATH/$DATASET/test_icd10_codes.txt \
--vocab-path $DATA_PATH/$DATASET/vocab10.npy
