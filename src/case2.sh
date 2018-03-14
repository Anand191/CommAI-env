#! /bin/sh

DATA_PATH=Seq2Seq_Attn/modular_reversed/data
PLOT_PATH=Seq2Seq_Attn/modular_reversed/Infer_Results
EPOCHS=100
EMBEDDING=300
HIDDEN=300

#Start Training
echo "\n\n Training For Case 2: Final + Intermediate"
python main.py --data $DATA_PATH --infer $PLOT_PATH --epochs $EPOCHS --print_every 20 --plot_every 10 --embedding_size $EMBEDDING --hidden_size $HIDDEN  --use_interim