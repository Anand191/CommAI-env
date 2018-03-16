#! /bin/sh

DATA_PATH=Seq2Seq_Attn/modular_reversed/data
PLOT_PATH=Seq2Seq_Attn/modular_reversed/Infer_Results
EPOCHS=500
EMBEDDING=300
HIDDEN=300

#Start Training
echo "\n\n Training For Case 4: Final + Attn_Guidance + Copy"
python main.py --data $DATA_PATH --infer $PLOT_PATH --epochs $EPOCHS --print_every 20 --plot_every 10 --embedding_size $EMBEDDING --hidden_size $HIDDEN --use_copy --use_attn --lr 0.01 --clip 5.0