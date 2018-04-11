#! /bin/sh

DATA_PATH=Seq2Seq_Attn/batchified/data
PLOT_PATH=Seq2Seq_Attn/batchified/Infer_Results/Model
ENCODER_WEIGHTS=Seq2Seq_Attn/batchified/Encoder/Model
DECODER_WEIGHTS=Seq2Seq_Attn/batchified/Decoder/Model

#Start Testing
echo "\n\n Running Inference for Hardcoded Attention"
python run_infer.py --data $DATA_PATH --infer $PLOT_PATH --encoder_weights $ENCODER_WEIGHTS --decoder_weights $DECODER_WEIGHTS  --use_copy
