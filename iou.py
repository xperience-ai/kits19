# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import data
import numpy as np


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids >= 0
    return np.bincount(pred_ids[idxs] * data.num_cl + gt_ids[idxs], minlength=data.num_cl * data.num_cl). \
        reshape((data.num_cl, data.num_cl)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [l for l in data.VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)


def evaluate(pred_ids, gt_ids):
    print('Evaluating', gt_ids.size, 'points...')
    confusion = confusion_matrix(pred_ids, gt_ids)
    class_ious = {}
    mean_iou = 0
    for i in range(data.num_cl):
        label_name = data.CLASS_LABELS[i]
        label_id = data.VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)
        mean_iou += class_ious[label_name][0] / data.num_cl

    print('classes          IoU')
    print('----------------------------')
    for i in range(data.num_cl):
        label_name = data.CLASS_LABELS[i]
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0],
                                                               class_ious[label_name][1],
                                                               class_ious[label_name][2]))
    print('mean IOU = {:.3f}\n'.format(mean_iou))
    return class_ious
