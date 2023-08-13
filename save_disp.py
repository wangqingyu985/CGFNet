from __future__ import print_function, division
import argparse
import os
import cv2
import torch.backends.cudnn as cudnn
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='CGFNet')
parser.add_argument('--model', default='cgfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='middlebury14', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/media/wangqingyu/机械硬盘2/立体匹配公开数据集/01_middlebury数据集/middlebury2014', help='data path')
parser.add_argument('--testlist', default='/media/wangqingyu/机械硬盘1/##model/CGFNet/filenames/middlebury14_train.txt', help='testing list')
parser.add_argument('--loadckpt', default='/media/wangqingyu/机械硬盘1/##model/CGFNet/1:1/sf.ckpt', help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=1, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, False, False)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test():
    os.makedirs(name='./c_t_m', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            h, w = disp_est.shape
            disp = np.array(disp_est[top_pad + 5:h, 0:w - right_pad-10], dtype=np.float32)
            fn = os.path.join("c_t_m", fn.split('/')[-2])
            fn += '.png'
            print("saving to", fn, disp.shape)

            disp = disp / np.max(disp) * 255
            disp_est_color = disp.astype(np.uint8)
            disp_est_color = cv2.applyColorMap(src=disp_est_color, colormap=cv2.COLORMAP_TURBO)
            cv2.imwrite(filename=fn, img=disp_est_color)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
