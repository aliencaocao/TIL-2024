# from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc


from statistics import mean
from typing import List


def bb_iou(bb1: List[int], bb2: List[int]) -> int:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in ltwh format.

    Parameters
    ----------
    bb1 : list[int, int, int, int]
        left, top, width, height
    bb2 : list[int, int, int, int]
        left, top, width, height

    Returns
    -------
    int
        0 or 1
    """
    boxA = [bb1[0], bb1[1], bb1[0] + bb1[2], bb1[1] + bb1[3]]
    boxB = [bb2[0], bb2[1], bb2[0] + bb2[2], bb2[1] + bb2[3]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0.0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value @ 0.5
    return round(iou)


def vlm_eval(bbox_truths: List[List[int]], bbox_predictions: List[List[int]]) -> float:
    return mean(
        bb_iou(bb_truth, bb_pred)
        for bb_truth, bb_pred in zip(bbox_truths, bbox_predictions)
    )
