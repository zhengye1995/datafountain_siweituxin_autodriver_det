
# 2019 第五届“四维图新”杯创新大赛 [自动驾驶视觉综合感知算法赛](https://www.datafountain.cn/competitions/366) detection baseline

**testA map**: 0.2

**author**： zhengye

for the semantic segmentation part, please refer to my teammate's git repo: https://github.com/zhengye1995/datafountain_siweituxin_autodriver_det

## 环境配置

请按照[mmdetection](https://github.com/open-mmlab/mmdetection)说明进行安装配置

### 训练

- **数据准备**

  将训练数据所有训练图片放置于 data/siweituxin/train_image
  
  两批次的数据label分别位于'data/dataset1/train.txt' 和 'data/DF_1018/train.txt'
  
  合并label： python tools/convert_datasets/merage_txt_label.py
  
  txt转coco json： python  tools/convert_datasets/trans_txt2json.py

- **模型训练**

  请依据mmdetection说明依据自身显卡情况线性调整lr，这里以4卡为例
  
  CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh config/siweituxin/faster_rcnn_r50_fpn.py 4


### 预测

- **数据准备**

  将测试数据所有训练图片放置于 data/siweituxin/test_images

- **inference**
  
  CUDA_VISIBLE_DEVICES=0 python tools/infer_siweituxin.py config/siweituxin/faster_rcnn_r50_fpn.py /work_dirs/faster_rcnn_r50_fpn/latest.pth --out det.txt


## Contact

This repo is currently maintained by Ye Zheng ([@zhengye1995](https://github.com/zhengye1995)).