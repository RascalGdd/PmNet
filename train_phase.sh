export CC=gcc-11
export CXX=g++-11

python -m torch.distributed.launch \
--nproc_per_node=2 \
run_phase_training.py \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model timesformer \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 3e-5 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /home/diandian/Diandian/DD/PmNet/data/PmLR50 \
--eval_data_path /home/diandian/Diandian/DD/PmNet/data/PmLR50 \
--nb_classes 5 \
--data_strategy online \
--output_mode key_frame \
--num_frames 20 \
--sampling_rate 8 \
--data_set LungSeg \
--data_fps 1fps \
--output_dir /home/diandian/Diandian/DD/PmNet/pmlr_results \
--log_dir /home/diandian/Diandian/DD/PmNet/pmlr_results \
--num_workers 10 \
--enable_deepspeed \
--no_auto_resume \
--dist_eval