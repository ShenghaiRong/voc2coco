# voc2coco
This is script for converting VOC instance annotations(ins_gt_mask.png) to COCO format json(ex. coco_style.json).

The script also supports converting VOC format xml or npy to COCO format json.

Owing to including the instance segmentation annotation, the voc_cocostyle.json can be used to train the instance segmentation network(e.g. Mask R-CNN).

We use [MMDetection](https://github.com/open-mmlab/mmdetection) to train Mask R-CNN with the generated voc2012_train_aug_cocostyle.json.

|    Dataset      |    Backbone     | Lr schd |  box mAP50 | mask mAP50 |
| :-------------: | :-------------: | :-----: | :----: | :-----: |
|  VOC2012-Val |R-50-FPN|   2x   | 73.9   | 67.3   |  

## Run script

```bash
$ python voc2coco.py \
    --image_list_dir /path/to/image_list/dir \
    --image_dir /path/to/image/dir \
    --ann_dir /path/to/annotation/dir \
    --labels /path/to/labels.txt \
    --output /path/to/output.json \
    --voc_ins_label_style 'png' \
    --seg_dir /path/to/GTsegmentation/dir \
```

## Check cocostyle json
As shown in [test.ipynb](./test.ipynb), we provide a script to check the generated json file.

## Download cocostyle json
We also directly provide the ground truth cocostyle json file of voc2012 for downloading.

Download link: https://pan.baidu.com/s/1_20wk1uFxKo6VvKEttCkwg?pwd=77ki Code: 77ki
