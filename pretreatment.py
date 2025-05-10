import os
import cv2
import numpy as np
import argparse
import imageio
from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
from time import sleep
from warnings import warn

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
parser.add_argument('--worker_num', default=1, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    if isinstance(logs, str):
        logs = [logs]
    return ''.join([f"# JOB {pid} : --{comment}-- {log}\n" for log in logs])


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH] and pid % 500 == 0:
        print(str_log, end='')


def cut_img(img, seq_info, frame_name, pid):
    if np.sum(img) <= 10000:
        message = f"seq:{'-'.join(seq_info)}, frame:{frame_name}, no data, {np.sum(img)}."
        warn(message)
        log_print(pid, WARNING, message)
        return None

    y = np.sum(img, axis=1)
    y_top = np.argmax(y != 0)
    y_btm = np.argmax(y[::-1] != 0)
    img = img[y_top:img.shape[0] - y_btm, :]

    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)

    sum_point = np.sum(img)
    sum_column = np.cumsum(np.sum(img, axis=0))
    x_center = np.searchsorted(sum_column, sum_point / 2)

    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W

    if left < 0 or right > img.shape[1]:
        padding = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([padding, img, padding], axis=1)
        left += h_T_W
        right += h_T_W

    return img[:, left:right].astype('uint8')


def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    os.makedirs(out_dir, exist_ok=True)

    frame_list = sorted(os.listdir(seq_path))
    count_frame = 0
    for frame_name in frame_list:
        frame_path = os.path.join(seq_path, frame_name)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        img = cut_img(img, seq_info, frame_name, pid)
        if img is not None:
            imageio.imwrite(os.path.join(out_dir, frame_name), img)
            count_frame += 1

    if count_frame < 5:
        message = f"seq:{seq_name}, less than 5 valid data."
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH, f"Contain {count_frame} valid frames. Saved to {out_dir}.")


if __name__ == "__main__":
    pool = Pool(WORKERS)
    results = []
    pid = 0

    print(f"Pretreatment Start.\nInput path: {INPUT_PATH}\nOutput path: {OUTPUT_PATH}\nLog file: {LOG_PATH}\nWorker num: {WORKERS}")

    for _id in sorted(os.listdir(INPUT_PATH)):
        for _seq_type in sorted(os.listdir(os.path.join(INPUT_PATH, _id))):
            for _view in sorted(os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))):
                seq_info = [_id, _seq_type, _view]
                out_dir = os.path.join(OUTPUT_PATH, *_id, _seq_type, _view)
                os.makedirs(out_dir, exist_ok=True)
                results.append(pool.apply_async(cut_pickle, args=(seq_info, pid)))
                pid += 1
                sleep(0.02)

    pool.close()
    pool.join()
    print("✅ Pretreatment hoàn tất!")
