#! /bin/sh

DATA_PATH=Seq2Seq_Attn/modular_reversed/data
PLOT_PATH=Seq2Seq_Attn/modular_reversed/Infer_Results

#Start Training
echo "\n\n Training"
python main.py --data $DATA_PATH --infer $PLOT_PATH