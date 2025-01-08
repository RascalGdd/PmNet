import os
import cv2
import pickle
from tqdm import tqdm
import random

def main():
    ROOT_DIR = "C:\DD\PmNet\data\PmLR50"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])
    VIDEO_NUMBERS = len(VIDEO_NAMES)
    INFER_NUMBERS = list(range(38, 48))

    INFER_FRAME_NUMBERS = 0

    infer_pkl = dict()

    unique_id = 0
    unique_id_infer = 0

    for video_id in VIDEO_NAMES:
        vid_id = int(video_id)

        if vid_id in INFER_NUMBERS:
            unique_id = unique_id_infer

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)
        frames_list = sorted([x for x in frames_list if "jpg" in x])

        # 打开Label文件
        anno_path = os.path.join(ROOT_DIR, 'phase_annotations', video_id + '.txt')
        anno_file = open(anno_path, 'r')
        anno_results = anno_file.readlines()[1:]
        print(len(frames_list))
        print(len(anno_results))
        assert len(frames_list) == len(anno_results)
        frame_infos = list()
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id
            info['video_id'] = video_id
            info['frames'] = len(frames_list)
            anno_info = anno_results[frame_id]
            anno_frame = anno_info.split()[0]
            assert int(anno_frame) == frame_id
            anno_id = anno_info.split()[1]
            info['phase_gt'] = int(anno_id)
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1

        if vid_id in INFER_NUMBERS:
            infer_pkl[video_id] = frame_infos
            INFER_FRAME_NUMBERS += len(frames_list)
            unique_id_infer = unique_id


    infer_save_dir = os.path.join(ROOT_DIR, 'labels', 'infer')
    os.makedirs(infer_save_dir, exist_ok=True)
    with open(os.path.join(infer_save_dir, '1fpsinfer.pickle'), 'wb') as file:
        pickle.dump(infer_pkl, file)

    print('INFER Frams', INFER_FRAME_NUMBERS, unique_id_infer)

if __name__ == "__main__":
    main()
