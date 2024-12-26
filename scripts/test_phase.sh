export CC=gcc-11
export CXX=g++-11

CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 12324 \
downstream_phase/run_phase_training.py \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model surgformer_HTA \
--pretrained_path pretrain_params/timesformer_base_patch16_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /jhcnas4/syangcw/M2CAI16-workflow \
--eval_data_path /jhcnas4/syangcw/M2CAI16-workflow \
--nb_classes 8 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--eval \
--finetune /jhcnas4/syangcw/Surgformerv2/M2CAI16/surgformer_HTA_M2CAI16_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best/mp_rank_00_model_states.pt \
--data_set M2CAI16 \
--data_fps 1fps \
--output_dir /jhcnas4/syangcw/Surgformerv2/M2CAI16 \
--log_dir /jhcnas4/syangcw/Surgformerv2/M2CAI16 \
--num_workers 10 \
--dist_eval \
--enable_deepspeed \
--no_auto_resume