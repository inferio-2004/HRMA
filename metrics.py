import numpy as np
import torch
import torch.nn.functional as F

import SimpleITK as sitk
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
    
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy
    
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)
def hausdorff_distance(lT,lP):
    try:
        lP = np.squeeze(lP,axis=1).transpose(1,2,0)
        lT = np.squeeze(lT,axis=1).transpose(1,2,0)
        labelPred=sitk.GetImageFromArray(lP)
        labelTrue=sitk.GetImageFromArray(lT)
        hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
        hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
        return hausdorffcomputer.GetAverageHausdorffDistance()
    except:
        return 1
import cv2

def surface_distances(predict, target, spacing=(1.0, 1.0, 1.0)):
    """
    计算表面距离
    predict: 预测结果，0表示背景，1表示前景
    target: 目标结果，0表示背景，1表示前景
    spacing: 数据体的间隔，每个轴上像素之间的物理距离
    """
    assert predict.shape == target.shape
    assert len(predict.shape) == 3

    # 计算最近距离
    result = np.zeros_like(predict)
    surface = cv2.Canny(target.astype(np.uint8), 0, 1)
    for i in range(predict.shape[0]):
        result[i] = cv2.distanceTransform(surface[i], cv2.DIST_L2, 3)

    # 计算平均表面距离
    intersect = np.count_nonzero(np.logical_and(predict, target))
    if intersect == 0:
        return 0
    else:
        return np.sum(result * np.logical_and(predict, target)) / intersect

def asd(predict, target, spacing=(1.0, 1.0, 1.0)):
    """
    计算ASD
    predict: 预测结果，0表示背景，1表示前景
    target: 目标结果，0表示背景，1表示前景
    spacing: 数据体的间隔，每个轴上像素之间的物理距离
    """
    predict = np.squeeze(predict,axis=1).transpose(1,2,0)
    target = np.squeeze(target,axis=1).transpose(1,2,0)
    return (surface_distances(predict, target, spacing) + surface_distances(target, predict, spacing)) / 2.0
