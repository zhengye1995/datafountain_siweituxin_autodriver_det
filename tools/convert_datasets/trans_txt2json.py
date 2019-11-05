# *utf-8*
import os
import json
import numpy as np
from tqdm import tqdm
import cv2

defect_name2label = {
    'red': 1, 'green': 2, 'yellow': 3, 'red_left': 4, 'red_right': 5, 'yellow_left': 6, 'yellow_right': 7,
    'green_left': 8, 'green_right': 9, 'red_forward': 10, 'green_forward': 11, 'yellow_forward': 12,
    'horizon_red': 13, 'horizon_green': 14, 'horizon_yellow': 15, 'off': 16, 'traffic_sign': 17,
    'car': 18, 'motor': 19, 'bike': 20, 'bus': 21, 'truck': 22, 'suv': 23, 'express': 24, 'person': 25,
}


class Siwei2COCO:

    def __init__(self, mode="train"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode = mode


    def to_coco(self, anno_file, img_dir):
        self._init_categories()
        with open(anno_file, 'r') as f:
            annos = f.readlines()

        for anno in tqdm(annos):
            try:
                img_name, seg_name, bboxs = anno.strip().split(' ', 2)
            except:
                img_name, seg_name = anno.strip().split(' ', 2)
                print(img_name)
                continue

            bboxs = bboxs.split(' ')
            # print(bboxs)

            img_path = os.path.join(img_dir, img_name)
            # img = cv2.imread(img_path)
            # h, w, _ = img.shape
            h, w = 720, 1280
            self.images.append(self._image(img_path, h, w))
            for bbox in zip(bboxs):
                # print(list(bbox)[0])
                xmin, ymin, xmax, ymax, class_id, _ = list(bbox)[0].split(',')
                # print(xmin, ymin, xmax, ymax, class_id)
                annotation = self._annotation(class_id, [float(xmin), float(ymin), float(xmax), float(ymax)], h, w)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        # for v in range(1, 16):
        # print(v)
        # category = {}
        # category['id'] = v
        # category['name'] = str(v)
        # category['supercategory'] = 'defect_name'
        # self.categories.append(category)
        for k, v in defect_name2label.items():
            category = {}
            category['id'] = v
            category['name'] = k
            category['supercategory'] = 'siweituxin_name'
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, h, w):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # area=abs(bbox[2]-bbox[0])*abs(bbox[3]-bbox[1])
        if area <= 0:
            print(bbox)
            input()
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(label)
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points, h, w)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _get_box(self, points, img_h, img_w):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        w = max_x - min_x
        h = max_y - min_y
        if w > img_w:
            w = img_w
        if h > img_h:
            h = img_h
        return [min_x, min_y, w, h]

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))


'''转换有瑕疵的样本为coco格式'''
img_dir = "data/siweituxin/train_image"
anno_dir = "data/siweituxin/Annotations/train.txt"
siwei2coco = Siwei2COCO()
train_instance = siwei2coco.to_coco(anno_dir, img_dir)

siwei2coco.save_coco_json(train_instance,
                           "data/siweituxin/annotations/"
                           + 'instances_{}.json'.format("train"))