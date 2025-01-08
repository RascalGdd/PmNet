# [AAAI 2025] Surgical Workflow Recognition and Blocking Effectiveness Detection in Laparoscopic Liver Resections with Pringle Maneuver


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Abstract

> Pringle maneuver (PM) in laparoscopic liver resection aims to reduce blood loss and provide a clear surgical view by intermittently blocking blood inflow of the liver, whereas prolonged PM may cause ischemic injury. To comprehensively monitor this surgical procedure and provide timely warnings of ineffective and prolonged blocking, we suggest two complementary AI-assisted surgical monitoring tasks: workflow recognition and blocking effectiveness detection in liver resections. The former presents challenges in real-time capturing of short-term PM, while the latter involves the intraoperative discrimination of long-term liver ischemia states. To address these challenges, we meticulously collect a novel dataset, called PmLR50, consisting of 25,037 video frames covering various surgical phases from 50 laparoscopic liver resection procedures. Additionally, we develop an online baseline for PmLR50, termed PmNet. This model embraces Masked Temporal Encoding (MTE) and Compressed Sequence Modeling (CSM) for efficient short-term and long-term temporal information modeling, and embeds Contrastive Prototype Separation (CPS) to enhance action discrimination between similar intraoperative operations. Experimental results demonstrate that PmNet outperforms existing state-of-the-art surgical workflow recognition methods on the PmLR50 benchmark. Our research offers potential clinical applications for the laparoscopic liver surgery community.


## 🔥🔥🔥 News!!
* Dec 12, 2024: 🤗 Our work has been accepted by AAAI 2025! Congratulations!
* Dec 26, 2024: 🚀 Code for General Surgical Workflow Recognition has been released!

## 📑 Open-source Plan

- General Surgical Workflow Recognition
  - [x] PmLR50 Dataset
  - [x] Pmnet

- Blocking Effectiveness Detection
  - [x] PmLR50 Dataset (Bounding boxes)

## General Surgical Workflow Recognition
### Installation
* Environment: CUDA 11.4 / Python 3.8
* Device: Two NVIDIA GeForce RTX 4090s
* Create a virtual environment
```shell
> conda env create -f PmNet.yml
```
### Prepare your data
Download processed data from [PmLR50(testset)](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155229775_link_cuhk_edu_hk/ERLDpgFVWvhDp_mFxl5xnZkBy822enSjvkT_TTpyvKJLog?e=bNkMlS);
The final structure of datasets should be as following:

```bash
data/
    └──PmLR50/
        └──frames/
            └──01
                ├──00000.png
                ├──00001.png
                └──...
            ├──...    
            └──50
        └──labels_pkl/
            └──train
            ├──val
            └──test
```
### Training
We provide the script for training [train_phase.sh](https://github.com/RascalGdd/PmNet/blob/main/train_phase.sh).

run the following code for training

```shell
sh scripts/train_phase.sh
```
> You need to modify **data_path**, **eval_data_path**, **output_dir** and **log_dir** according to your own setting.

### Test
> Currently, the test and evaluation codes we provide are only applicable to two-GPU inference.

1. run the following code for testing, and get **0.txt** and **1.txt**;

```shell
sh scripts/test.sh
```
### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|       `--batch_size`       |   8    |   The batch size for training and inference   |
|     `--epochs`     | 50  |      The max epoch for training      |
|    `--save_ckpt_freq`    |    10    |     The frequency for saving checkpoints during training     |
|    `--nb_classes`     |    5     |     The number of classes for surgical workflows      |
| `--data_strategy` |    online    |    Online/offline mode       |
|     `--num_frames`     |    20    | The number of consecutive frames used  |
|     `--sampling_rate`   |    8  | The sampling interval for comsecutive frames |
|        `--enable_deepspeed`        |    True  |   Use deepspeed to accelerate  |
|  `--dist_eval`   |   True   |    Use distributed evaluation to accelerate    |

### Checkpoint

The checkpoint of our model is provided [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155229775_link_cuhk_edu_hk/EaymSNkgnLJJry1PqwINcVUBQzBYPt53fE9c574Z08TWSg?e=4IWBKO).

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [TMRNet](https://github.com/YuemingJin/TMRNet)
- [Surgformer](https://github.com/isyangshu/Surgformer/)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)

## License & Citation 
If you find our work useful in your research, please consider citing our paper.
