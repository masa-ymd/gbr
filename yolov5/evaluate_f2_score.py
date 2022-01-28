import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

# val bbox result directory
LABEL_DIR = '/root/kaggle/tensorflow-great-barrier-reef/data/yolo_data/fold1/labels/val'
PRD_BBOX_DIR = '/root/kaggle/yolov5/runs/val/exp/labels/'

def calc_iou(bboxes1, bboxes2, bbox_mode='xywh'):
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4
    
    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()
    
    if bbox_mode == 'xywh':
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def f_beta(tp, fp, fn, beta=2):
    return (1+beta**2)*tp / ((1+beta**2)*tp + beta**2*fn+fp)

def calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()
    
    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes): # fixed in ver.7
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou > iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - (k + 1) # fix in ver.7
            break

    fn = len(gt_bboxes)
    return tp, fp, fn

def calc_is_correct(gt_bboxes, pred_bboxes):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns
    
    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes)*11, 0
        return tps, fps, fns
    
    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)*11
        return tps, fps, fns
    
    pred_bboxes = pred_bboxes[pred_bboxes[:,0].argsort()[::-1]] # sort by conf
    
    tps, fps, fns = 0, 0, 0
    for iou_th in np.arange(0.3, 0.85, 0.05):
        tp, fp, fn = calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns

def calc_f2_score(gt_bboxes_list, pred_bboxes_list, verbose=False):
    """
    gt_bboxes_list: list of (N, 4) np.array in xywh format
    pred_bboxes_list: list of (N, 5) np.array in conf+xywh format
    """
    tps, fps, fns = 0, 0, 0
    for gt_bboxes, pred_bboxes in zip(gt_bboxes_list, pred_bboxes_list):
        tp, fp, fn = calc_is_correct(gt_bboxes, pred_bboxes)
        tps += tp
        fps += fp
        fns += fn
        if verbose:
            num_gt = len(gt_bboxes)
            num_pred = len(pred_bboxes)
            print(f'num_gt:{num_gt:<3} num_pred:{num_pred:<3} tp:{tp:<3} fp:{fp:<3} fn:{fn:<3}')
    return f_beta(tps, fps, fns, beta=2)

paths = glob(f'{LABEL_DIR}/*')
val_len = len(paths) 
print(f'made bounding box of {len(os.listdir(PRD_BBOX_DIR))} images in validation set ')

gt_bboxs_list, prd_bboxs_list = [], []
count=0
"""
for image_file in paths:
    gt_bboxs = []
    prd_bboxs = []
    with open(image_file, 'r') as f:
        while True:
            r = f.readline().rstrip()
            if not r: break
            r = r.split()[1:]
            bbox = np.array(list(map(float, r))); gt_bboxs.append(bbox)

    pred_path = f'{PRD_BBOX_DIR}/'
    pred_file = pred_path+image_file[27:]

    no_anns = True
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            while True:
                r = f.readline().rstrip()
                if not r: break
                r = r.split()[1:]; r = [r[4], *r[:4]]
                conf=float(r[0])
                #if conf>confidence: 
                bbox = np.array(list(map(float, r)))
                prd_bboxs.append(bbox)
                no_anns = False

    if no_anns: count+=1

    gt_bboxs, prd_bboxs= np.array(gt_bboxs), np.array(prd_bboxs)
    prd_bboxs_list.append(prd_bboxs)
    gt_bboxs_list.append(gt_bboxs)
"""

for image_file in paths:
    print(image_file)
    #txt_name = image_file[:-4]+'.txt'
    gt_bboxs = []
    prd_bboxs = []
    with open(image_file, 'r') as f:
        while True:
            r = f.readline().rstrip()
            if not r:
                break
            r = r.split()[1:]
            bbox = np.array(list(map(float, r)))
            gt_bboxs.append(bbox)

    f = os.path.splitext(os.path.basename(image_file))[0]
    print(f)

    if os.path.exists(PRD_BBOX_DIR+f):
        with open(PRD_BBOX_DIR+txt_name, 'r') as f:
            while True:
                r = f.readline().rstrip()
                if not r:
                    break
                r = r.split()[1:]
                r = [r[4], *r[:4]]
                bbox = np.array(list(map(float, r)))
                prd_bboxs.append(bbox)
    gt_bboxs, prd_bboxs = np.array(gt_bboxs), np.array(prd_bboxs)
    gt_bboxs_list.append(gt_bboxs)
    prd_bboxs_list.append(prd_bboxs)
    count += 1
print(f'{count} bound boxs appended to list')



print(f'{count} bound boxs appended to list')

score = calc_f2_score(gt_bboxs_list, prd_bboxs_list, verbose=True)
print(f'f2 score for validation set is {score}')