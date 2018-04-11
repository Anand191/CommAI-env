#! /bin/sh

DATA_PATH=Seq2Seq_Attn/batchified/data
PLOT_PATH=Seq2Seq_Attn/batchified/Infer_Results/Model
ENCODER_WEIGHTS=Seq2Seq_Attn/batchified/Encoder/Model
DECODER_WEIGHTS=Seq2Seq_Attn/batchified/Decoder/Model
EPOCHS=200
EMBEDDING=300
HIDDEN=300

#Start Training
echo "\n\n Training For Case 4: Final + Attn_Guidance + Copy"
python main_batch.py --data $DATA_PATH --infer $PLOT_PATH --encoder_weights $ENCODER_WEIGHTS --decoder_weights $DECODER_WEIGHTS --epochs $EPOCHS --print_every 25 --plot_every 10 --test_every 100 --embedding_size $EMBEDDING --hidden_size $HIDDEN --use_copy  --lr 0.01 --clip 5.0
