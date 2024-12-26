export CC=gcc-11
export CXX=g++-11

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12326 \
downstream_phase/run_phase_training.py \
--batch_size 8 \
--epochs 50 \
--save_ckpt_freq 10 \
--model timesformer \
--pretrained_path pretrain_params/timesformer_base_patch16_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 5 \
--data_path /home/syangcw/LungSeg/LungSeg \
--eval_data_path /home/syangcw/LungSeg/LungSeg \
--nb_classes 7 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--data_set LungSeg \
--data_fps 1fps \
--output_dir /results/LungSeg \
--log_dir /results/LungSeg \
--num_workers 10 \
--dist_eval \
--enable_deepspeed \
--no_auto_resume