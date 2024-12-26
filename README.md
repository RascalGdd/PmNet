Official repository for Surgical Workflow Recognition and Blocking Effectiveness Detection in Laparoscopic Liver Resections with Pringle Maneuver.


## 🔥🔥🔥 News!!
* Dec 12, 2024: 🤗 Our work has been accepted by AAAI 2025! Congratulations!

## 📑 Open-source Plan

- General Surgical Workflow Recognition
  - [ ] PmLR50 Dataset
  - [x] Pmnet

- Surgical Workflow Recognition and Blocking Effectiveness Detection
  - [ ] PmLR50 Dataset (Bounding boxes)

## General Surgical Workflow Recognition

### Prepare your data
Download raw video data from [PmLR50](link);
The final structure of datasets should be as following:

Note that you can change `fps` in the Step.2 to generate more frames.
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
sh scripts/train.sh
```
> You need to modify **data_path**, **eval_data_path**, **output_dir** and **log_dir** according to your own setting.

### Test
> Currently, the test and evaluation codes we provide are only applicable to two-GPU inference.

1. run the following code for testing, and get **0.txt** and **1.txt**;

```shell
sh scripts/test.sh
```

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [TMRNet](https://github.com/YuemingJin/TMRNet)
- [Surgformer](https://github.com/isyangshu/Surgformer/)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)
