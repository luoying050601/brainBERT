base_dir='/project/brain-bert'
export BERT_BASE_DIR=$base_dir/original_data/text
export DATA_DIR=$base_dir/src/com/sakura/train/text/glue_data
start=$(date +%s)


echo $DATA_DIR
echo $BERT_BASE_DIR


python3 run_classifier.py \
 --task_name=QNLI \
 --do_train=true \
 --do_eval=true \
 --data_dir=$DATA_DIR/QNLI \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
 --max_seq_length=128 \
 --train_batch_size=32 \
 --learning_rate=2e-5 \
 --num_train_epochs=3.0 \
 --output_dir=$base_dir/output/BERT
end=$(date +%s)
take=$(( end - start ))
echo Time taken to execute commands is ${take} seconds.