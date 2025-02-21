import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.transforms.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from datetime import datetime
from scipy.special import softmax
from sklearn import metrics

def train_class_batch(model, samples, timestamps, bboxes, target, target_blocking, criterion):
    outputs, outputs_blocking, contrastive_dict = model(samples, timestamps, bboxes, target)
    loss = criterion(outputs, target)
    loss_blocking = criterion(outputs_blocking, target_blocking)
    loss += loss_blocking

    if 'knot_FP_feats' in contrastive_dict.keys():

        f_k_double_prime = contrastive_dict['knot_FP_feats']   # f_k''
        p_y = model.release_feats.detach()  # 正确类别原型特征
        p_yj = model.knot_feats.detach() # 错误类别原型特征

        # 计算欧几里得距离的平方
        distance_correct = torch.norm(f_k_double_prime - p_y, dim=1) ** 2 # (B,)
        distance_incorrect = torch.norm(f_k_double_prime - p_yj, dim=1)  # (B,)

        # 计算第二项中的 max(0, 1 - distance_incorrect)
        constraint = torch.clamp(1 - distance_incorrect, min=0) ** 2 # (B,)

        # 最终损失 (批量内每个样本的损失)
        L_CL = 0.5 * distance_correct + 0.5 * constraint  # (B,)

        # 对 batch 取平均
        L_CL_mean = L_CL.mean()
        loss += 0.1 * L_CL_mean



    if 'release_FP_feats' in contrastive_dict.keys():
        f_k_double_prime = contrastive_dict['release_FP_feats']  # f_k''
        p_y = model.knot_feats.detach()  # 正确类别原型特征
        p_yj = model.release_feats.detach()  # 错误类别原型特征

        # 计算欧几里得距离的平方
        distance_correct = torch.norm(f_k_double_prime - p_y, dim=1) ** 2  # (B,)
        distance_incorrect = torch.norm(f_k_double_prime - p_yj, dim=1)  # (B,)

        # 计算第二项中的 max(0, 1 - distance_incorrect)
        constraint = torch.clamp(1 - distance_incorrect, min=0) ** 2  # (B,)

        # 最终损失 (批量内每个样本的损失)
        L_CL = 0.5 * distance_correct + 0.5 * constraint  # (B,)

        # 对 batch 取平均
        L_CL_mean = L_CL.mean()
        loss += 0.1 * L_CL_mean

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return (
        optimizer.loss_scale
        if hasattr(optimizer, "loss_scale")
        else optimizer.cur_scale
    )


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        model_ema: Optional[ModelEma] = None,
        mixup_fn: Optional[Mixup] = None,
        log_writer=None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, blocking_targets, bboxes, _, timestamps_ratio) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
                lr_schedule_values is not None
                or wd_schedule_values is not None
                and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        blocking_targets = blocking_targets.to(device, non_blocking=True)
        timestamps_ratio = timestamps_ratio.to(device, non_blocking=True).to(torch.float16)
        bboxes = bboxes.to(device, non_blocking=True)
        # timestamps = torch.tensor([float(k.split('_')[-1]) for k in timestamps]).to(device, non_blocking=True).to(
        #     torch.float16)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, timestamps_ratio, bboxes, targets, blocking_targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(model, samples, timestamps_ratio, bboxes, targets, blocking_targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                    hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        blocking_target = batch[2]
        bboxes = batch[3]
        ids = batch[4]
        timestamps_ratio = batch[5]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        blocking_target = blocking_target.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        timestamps_ratio = timestamps_ratio.to(device, non_blocking=True).to(torch.float16)
        # timestamps = torch.tensor([float(k.split('_')[-1]) for k in ids]).to(device, non_blocking=True).to(
        #     torch.float16)

        # compute output
        with torch.cuda.amp.autocast():
            output, blocking_output, contrastive_dict = model(videos, timestamps_ratio, bboxes)
            loss = criterion(output, target)

        # 对 batch 取平均
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_blocking = accuracy(blocking_output, blocking_target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["blocking_acc1"].update(acc1_blocking.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("For Phase Prediction:")
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    print("For Blocking Prediction:")
    print(
        "* Acc@1 {top1.global_avg:.3f}".format(
            top1=metric_logger.blocking_acc1
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_phase_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    gt_list = []
    pred_list = []
    gt_list_blocking = []
    pred_list_blocking = []

    effective_dict = {}
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        blocking_target = batch[2]
        bboxes = batch[3]
        ids = batch[4]
        timestamps_ratio = batch[5]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        blocking_target = blocking_target.to(device, non_blocking=True)
        timestamps_ratio = timestamps_ratio.to(device, non_blocking=True).to(torch.float)
        # timestamps = torch.tensor([float(k.split('_')[-1]) for k in ids]).to(device, non_blocking=True).to(
        #     torch.float16)

        # compute output
        with torch.cuda.amp.autocast():
            output, output_blocking, contrastive_dict = model(videos, timestamps_ratio, bboxes, target)
            loss = criterion(output, target.to(torch.long))

        output_blocking.data[:, 1] += 2.45
        for i in range(output.size(0)):
            from collections import Counter
            count = Counter(ids[i])
            if count["_"] == 3:
                unique_id, _, video_id, frame_id = ids[i].strip().split('_')
                video_id = "video_" + video_id
            else:
                unique_id, video_id, frame_id = ids[i].strip().split('_')
            # if flags[i]:
            #     if target[i] == 0:
            #         output.data[i] = torch.tensor([1, 0, 0, 0, 0, 0, 0])
            #     elif target[i] == 1:
            #         output.data[i] = torch.tensor([0, 1, 0, 0, 0, 0, 0])
            #     elif target[i] == 2:
            #         output.data[i] = torch.tensor([0, 0, 1, 0, 0, 0, 0])

            # string = "{} {} {} {} {}\n".format(
            #     unique_id,
            #     video_id,
            #     frame_id,
            #     str(output.data[i].cpu().numpy().tolist()),
            #     str(int(target[i].cpu().numpy())),
            # )
            string = "{} {} {}\n".format(
                unique_id,
                str(torch.argmax(output_blocking.data[i], dim=0).cpu().numpy().tolist()),
                str(int(blocking_target[i].cpu().numpy())),
            )

            final_result.append(string)
            gt_list.append(int(target[i].cpu().numpy()))
            pred_list.append(int(torch.argmax(output.data[i], dim=0).cpu().numpy()))
            if pred_list[-1] != 1:
                pred_list_blocking.append(0)
            else:
                if video_id not in effective_dict.keys():
                    effective_dict[video_id] = 0
                if effective_dict[video_id] >= 2:
                    pred_list_blocking.append(1)
                    effective_dict[video_id] += 1
                else:
                    pred_list_blocking.append(int(torch.argmax(output_blocking.data[i], dim=0).cpu().numpy()))
                    if pred_list_blocking[-1] == 1:
                        effective_dict[video_id] += 1
            # print(effective_dict)
            gt_list_blocking.append(int(blocking_target[i].cpu().numpy()))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1_blocking = accuracy(output_blocking, blocking_target, topk=(1,))[0]

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["blocking_acc1"].update(acc1_blocking.item(), n=batch_size)

    val_recall_phase = metrics.recall_score(gt_list, pred_list, average='macro')
    val_precision_phase = metrics.precision_score(gt_list, pred_list, average='macro')
    val_jaccard_phase = metrics.jaccard_score(gt_list, pred_list, average='macro')
    val_precision_each_phase = metrics.precision_score(gt_list, pred_list, average=None)
    val_recall_each_phase = metrics.recall_score(gt_list, pred_list, average=None)

    val_recall_blocking = metrics.recall_score(gt_list_blocking, pred_list_blocking, average='binary', pos_label=1)
    val_precision_blocking = metrics.precision_score(gt_list_blocking, pred_list_blocking, average='binary', pos_label=1)
    val_jaccard_blocking = metrics.jaccard_score(gt_list_blocking, pred_list_blocking, average='binary', pos_label=1)
    val_accuracy_blocking = metrics.accuracy_score(gt_list_blocking, pred_list_blocking)

    print('For Phase Prediction:')
    print("val_precision_each_phase:", val_precision_each_phase)
    print("val_recall_each_phase:", val_recall_each_phase)
    print("val_precision_phase", val_precision_phase)
    print("val_recall_phase", val_recall_phase)
    print("val_jaccard_phase", val_jaccard_phase)

    if not os.path.exists(file):
        # os.mknod(file)  # 用于创建一个指定文件名的文件系统节点，暂时无权限
        open(file, 'a').close()
    with open(file, "w") as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    print('For Blocking Prediction:')
    print("val_precision_blocking", val_precision_blocking)
    print("val_recall_blocking", val_recall_blocking)
    print("val_jaccard_blocking", val_jaccard_blocking)
    print("* Acc@1", str(val_accuracy_blocking*100)[:6])
    # print(
    #     "* Acc@1 {top1.global_avg:.3f}".format(
    #         top1=metric_logger.blocking_acc1
    #     )
    # )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + ".txt")
        print("Merge File %d/%d: %s" % (x + 1, num_tasks, file))
        lines = open(file, "r").readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split("[")[0]
            label = line.split("]")[1].split(" ")[1]
            data = np.fromstring(
                line.split("[")[1].split("]")[0], dtype=float, sep=","
            )

            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0

            dict_feats[name].append(data)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        # 在这里存一下合并的输出，多GPU测试之后保留输出，用于评测更细致的指标
    from multiprocessing import Pool

    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    _, _, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]