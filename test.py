import numpy as np
import argparse
from datetime import datetime
from model.initialization_keras import initialization
from utils.evaluator_keras import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


# Khởi tạo mô hình
m = initialization(conf, test=opt.cache)[0]

# Load checkpoint
print(f"Loading the model of iteration {opt.iter}...")
m.load(opt.iter)
print("Transforming...")
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf['data'])
print(f"Evaluation complete. Cost: {datetime.now() - time}")

# In kết quả Rank-1 (bao gồm identical-view cases)
for i in range(1):
    print(f"===Rank-{i+1} (Include identical-view cases)===")
    print(f"NM: {np.mean(acc[0, :, :, i]):.3f},\tBG: {np.mean(acc[1, :, :, i]):.3f},\tCL: {np.mean(acc[2, :, :, i]):.3f}")

# In kết quả Rank-1 (không bao gồm identical-view cases)
for i in range(1):
    print(f"===Rank-{i+1} (Exclude identical-view cases)===")
    print(f"NM: {de_diag(acc[0, :, :, i]):.3f},\tBG: {de_diag(acc[1, :, :, i]):.3f},\tCL: {de_diag(acc[2, :, :, i]):.3f}")

# In kết quả Rank-1 cho từng góc
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print(f"===Rank-{i+1} of each angle (Exclude identical-view cases)===")
    print("NM:", de_diag(acc[0, :, :, i], True))
    print("BG:", de_diag(acc[1, :, :, i], True))
    print("CL:", de_diag(acc[2, :, :, i], True))

print("✅ Testing hoàn tất!")
