# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np, glob, torch.utils.data, multiprocessing as mp
import torch.nn as nn
import sparseconvnet as scn
import os

color_map = [(0, 0, 0), (255, 0, 0), (0, 0, 255)]

# dataset flags

# training
# original_full_ds = False
# tissue_kidney_full_ds = False
# tissue_kidney_tumor_small_ds = True
# inference_mode = False

# inference
original_full_ds = False
tissue_kidney_full_ds = False
tissue_kidney_tumor_small_ds = True
inference_mode = False  # set to True while producing small dataset


def save_to_obj(data, colors, labels, out_file, color_map=color_map):
    with open(out_file, 'w') as f:
        print("Saving input sample to {}".format(out_file))
        print("point number:")
        print("tissue:", (labels[labels == 0]).shape)
        print("kidney:", (labels[labels == 1]).shape, ", color:", color_map[1])
        if original_full_ds or tissue_kidney_tumor_small_ds:
            print("tumor:", (labels[labels == 2]).shape, ", color:", color_map[2])
        for i in range(data.shape[0]):
            label = labels[i]
            if label > 0:
                color = color_map[label]
            elif label == 0:
                color = np.array([colors[i][0], colors[i][1], colors[i][2]])
                if color.max() < 2:  # [-1..1] -> [0..255]
                    color = color.astype(np.float32)
                    color = ((color + 1) / 2 * 255).astype(np.int8)
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2],
                                               color[0], color[1], color[2]))


def train_point_transform(tbl):
    locs = []
    feats = []
    labels = []
    for idx, i in enumerate(tbl):
        if not inference_mode:
            (a, b, c), case = i
        else:
            (a, b, c, d), case = i
        if a.shape[1] > 3:  # temp fix for small dataset, remove 4th column
            a = a[:, 0:3]
        m = np.eye(3)
        m *= scale
        m[0][0] *= scale_z / scale  # scale z axis by scale_z
        a = np.matmul(a, m)
        m = a.min(0)
        M = a.max(0)
        offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) + np.clip(full_scale - M + m + 0.001, None,
                                                                             0)
        a += offset

        # augmentations
        prob_thresh = 0.5

        # 1. random intensity level
        if (np.random.random() > prob_thresh):
            intensity_factor = np.random.random() * 0.5 + 0.5  # from interval [0.5 .. 1]
            b *= intensity_factor

        # 2. random aspect ratio
        if (np.random.random() > prob_thresh):
            aspect_ratio = np.random.random() * 0.4 + 0.8  # from interval [0.8 .. 1.2]
            a *= aspect_ratio

        # 3. random low level intensity drop
        if (np.random.random() > prob_thresh):
            lower_intensity = np.random.random() * 0.5  # from interval [0 .. 0.5]
            idxs = b[:, 0] > lower_intensity
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]

        idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]

        # set max_points to fit to memory
        if a.shape[0] > max_points:
            idxs = np.random.choice(a.shape[0], max_points, replace=False)
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]

        if tissue_kidney_full_ds:  # 2 classes only
            c[c == 2] = 1

        a = torch.from_numpy(a).long()
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    return {'x': [locs, feats], 'y': labels.long(), 'id': tbl}


def val_point_transform(tbl):
    locs = []
    feats = []
    labels = []
    for idx, i in enumerate(tbl):
        if not inference_mode:
            (a, b, c), case = i
        else:
            (a, b, c, _), case = i
        if a.shape[1] > 3:  # temp fix for small dataset, remove 4th column
            a = a[:, 0:3]
        m = np.eye(3)
        m *= scale
        m[0][0] *= scale_z / scale  # scale z axis by scale_z
        a = np.matmul(a, m)
        m = a.min(0)
        M = a.max(0)
        offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) + \
                 np.clip(full_scale - M + m + 0.001, None, 0)
        a += offset

        idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]

        # set max_points to fit to memory
        if a.shape[0] > max_points:
            idxs = np.random.choice(a.shape[0], max_points, replace=False)
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]

        if tissue_kidney_full_ds:  # 2 classes only
            c[c == 2] = 1

        a = torch.from_numpy(a).long()
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    return {'x': [locs, feats], 'y': labels.long(), 'id': tbl, 'case': case}


def val_point_transform_with_coords(tbl):
    locs = []
    feats = []
    labels = []
    coords = []
    for idx, i in enumerate(tbl):
        if not inference_mode:
            (a, b, c), case = i
        else:
            (a, b, c, d), case = i
        if a.shape[1] > 3:  # temp fix for small dataset, remove 4th column
            a = a[:, 0:3]
        m = np.eye(3)
        m *= scale
        m[0][0] *= scale_z / scale  # scale z axis by scale_z
        a = np.matmul(a, m)
        m = a.min(0)
        M = a.max(0)
        offset = -m + np.clip(full_scale - M + m - 0.001, 0, None) + \
                 np.clip(full_scale - M + m + 0.001, None, 0)
        a += offset

        idxs = (a.min(1) >= 0) * (a.max(1) < full_scale)
        a = a[idxs]
        b = b[idxs]
        c = c[idxs]
        d = d[idxs]

        # set max_points to fit to memory
        if a.shape[0] > max_points:
            idxs = np.random.choice(a.shape[0], max_points, replace=False)
            a = a[idxs]
            b = b[idxs]
            c = c[idxs]
            d = d[idxs]

        if tissue_kidney_full_ds and not inference_mode:  # 2 classes only
            c[c == 2] = 1

        a = torch.from_numpy(a).long()
        locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
        feats.append(torch.from_numpy(b))
        labels.append(torch.from_numpy(c))
        coords.append(torch.from_numpy(d))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    coords = torch.cat(coords, 0)
    return {'x': [locs, feats], 'y': labels.long(), 'id': tbl, 'case': case, 'coords': coords}


class kits_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.names = []
        self.root_dir = root_dir
        for f in glob.glob(self.root_dir + '/*.pth'):
            self.names.append(f)

    def __len__(self):
        return (len(self.names))

    def __getitem__(self, index):
        res = torch.load(self.names[index], map_location='cpu')
        case = os.path.basename(self.names[index])
        return res, case


class Model(nn.Module):
    def __init__(self, dimension, full_scale, num_classes):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4)).add(  # orig mode=4
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(  # inL=1
            scn.UNet(dimension, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                     residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, num_classes)

    def forward(self, x):
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


# Options
# Elastic distortion
# blur0 = np.ones((3, 1, 1)).astype('float32') / 3
# blur1 = np.ones((1, 3, 1)).astype('float32') / 3
# blur2 = np.ones((1, 1, 3)).astype('float32') / 3

m = 16  # 16 or 32
residual_blocks = False  # True or False
block_reps = 1  # Conv block repetition factor: 1 or 2

scale = 1  # Voxel size = 1/scale
scale_z = scale / 2.0
val_reps = 1  # Number of test views, 1 or more
dimension = 3
full_scale = 1024  # Input field size

if original_full_ds:
    VALID_CLASS_IDS = np.array([0, 1, 2])
    CLASS_LABELS = ['tissue', 'kidney', 'tumor']
    batch_size = 2
    max_points = 1000000  # batch with 12e6 points consumes 8GB GPU memory for batch_size <= 4

elif tissue_kidney_tumor_small_ds:
    VALID_CLASS_IDS = np.array([0, 1, 2])
    CLASS_LABELS = ['tissue', 'kidney', 'tumor']
    batch_size = 10
    max_points = 1000000

elif tissue_kidney_full_ds:
    VALID_CLASS_IDS = np.array([0, 1])
    CLASS_LABELS = ['tissue', 'kidney']
    batch_size = 2
    max_points = 1000000

num_cl = len(CLASS_LABELS)
num_workers = mp.cpu_count()
