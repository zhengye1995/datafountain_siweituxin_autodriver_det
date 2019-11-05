import os
from mmdet.apis import init_detector, inference_detector
import mmcv
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.out\
        ('Please specify out path'
         'with the argument "--out"')

    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    with open(args.out, 'w') as f:
        for img in tqdm(glob('data/siweituxin/test_images/*.jpg')):
            result = inference_detector(model, img)
            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            im_path = os.path.basename(img)
            line_str = str(im_path) + ' ' + str(im_path.split('.')[0]+'.png')
            if len(bboxes) > 0:
                for j, bbox in enumerate(bboxes):
                    bbox_int = bbox.astype(np.int32)
                    label = labels[j] + 1
                    score = bbox[4]
                    line_str += ' ' + str(bbox_int[0]) + ',' + str(bbox_int[1]) + ',' + str(bbox_int[2]) + ',' + \
                                str(bbox_int[3]) + ',' + str(label) + ',' + str(score)
            line_str += '\n'
            f.write(line_str)


if __name__ == '__main__':
    main()

