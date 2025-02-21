import os
import cv2
import pickle
from tqdm import tqdm
import random
import json

def main():
    ROOT_DIR = "/home/diandian/Diandian/DD/PmNet/data/PmLR50"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "frames"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])
    VIDEO_NUMBERS = len(VIDEO_NAMES)
    TRAIN_NUMBERS = list(range(1, 34))
    TRAIN_NUMBERS.append(49)
    TRAIN_NUMBERS.append(50)
    print('For training:', TRAIN_NUMBERS)
    TEST_NUMBERS = list(range(34, 38))
    TEST_NUMBERS.append(48)
    print('For validation:', TEST_NUMBERS)
    INFER_NUMBERS = list(range(38, 48))
    print('For testing:', INFER_NUMBERS)

    total_frame_count = 0
    for video_id in VIDEO_NAMES:
        if int(video_id )in TRAIN_NUMBERS:
            video_path = os.path.join(ROOT_DIR, "frames", video_id)
            total_frame_count += len(os.listdir(video_path))
    average_frame_number = total_frame_count/len(TRAIN_NUMBERS)

    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0
    INFER_FRAME_NUMBERS = 0

    train_pkl = dict()
    test_pkl = dict()
    infer_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0
    unique_id_infer = 0

    for video_id in VIDEO_NAMES:
        vid_id = int(video_id)

        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test
        elif vid_id in INFER_NUMBERS:
            unique_id = unique_id_infer

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)
        frames_list = sorted([x for x in frames_list if "jpg" in x])

        # 打开Label文件
        anno_path = os.path.join(ROOT_DIR, 'phase_annotations', video_id + '.txt')
        anno_file = open(anno_path, 'r')
        anno_results = anno_file.readlines()[1:]
        blocking_path = os.path.join(ROOT_DIR, 'blocking_annotations', video_id + '.txt')
        blocking_file = open(blocking_path, 'r')
        blocking_results = blocking_file.readlines()[1:]
        bbox_path = os.path.join(ROOT_DIR, 'bbox_annotations', video_id + '.json')
        with open(bbox_path, 'r', encoding='utf-8') as f:
            bbox_data = json.load(f)
        assert len(frames_list) == len(anno_results)
        frame_infos = list()
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id
            info['video_id'] = video_id
            info['frames'] = len(frames_list)
            anno_info = anno_results[frame_id]
            blocking_info = blocking_results[frame_id]
            anno_frame = anno_info.split()[0]
            assert int(anno_frame) == frame_id
            anno_id = anno_info.split()[1]
            blocking_id = blocking_info.split()[1]
            info['phase_gt'] = int(anno_id)
            info['blocking_gt'] = int(blocking_id)
            origin_bbox = bbox_data[frames_list[frame_id].split('.')[0]]
            x1, x2, y1, y2 = origin_bbox[0][0], origin_bbox[1][0], origin_bbox[0][1], origin_bbox[1][1]
            if vid_id not in [16, 18, 46, 47, 48, 49]:
                x1, x2 = x1 * 224 / 1280, x2 * 224 / 1280
                y1, y2 = y1 * 224 / 720,  y2 * 224 / 720
            else:
                x1, x2 = x1 * 224 / 1920, x2 * 224 / 1920
                y1, y2 = y1 * 224 / 1080,  y2 * 224 / 1080

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(224, int(x2)), min(224, int(y2))
            info['bbox'] = [[x1, y1], [x2, y2]]
            info['fps'] = 1
            info['avg_length'] = average_frame_number
            frame_infos.append(info)
            unique_id += 1

        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frames_list)
            unique_id_test = unique_id
        elif vid_id in INFER_NUMBERS:
            infer_pkl[video_id] = frame_infos
            INFER_FRAME_NUMBERS += len(frames_list)
            unique_id_infer = unique_id

    train_save_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    infer_save_dir = os.path.join(ROOT_DIR, 'labels', 'infer')
    os.makedirs(infer_save_dir, exist_ok=True)
    with open(os.path.join(infer_save_dir, '1fpsinfer.pickle'), 'wb') as file:
        pickle.dump(infer_pkl, file)

    print('TRAIN Frames', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frames', TEST_FRAME_NUMBERS, unique_id_test)
    print('INFER Frames', INFER_FRAME_NUMBERS, unique_id_infer)
    print('Average Length', (TRAIN_FRAME_NUMBERS)/len(TRAIN_NUMBERS))

if __name__ == "__main__":
    main()
