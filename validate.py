# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import convert
import data
import iou
import numpy as np
import torch
import glob

color_map = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 0, 255)
]


def save_to_obj(points, colors, labels, out_file, color_map=color_map):
    with open(out_file, 'w') as f:
        print("Saving scene from input sample to {}".format(out_file))
        print("points number:")
        print("tissue:", (labels[labels == 0]).shape)
        print("kidney:", (labels[labels == 1]).shape, ", color:", color_map[1])
        if data.num_cl ==3:
            print("tumor:", (labels[labels == 2]).shape, ", color:", color_map[2])
        for i in range(points.shape[0]):
            label = labels[i]
            if label > 0:
                color = color_map[label]
            elif label == 0:
                color = np.array([colors[i][0], colors[i][1], colors[i][2]])
                if color.max() <= 1:  # [-1..1] -> [0..255]
                    color = color.astype(np.float32)
                    color = ((color + 1) / 2 * 255).astype(np.int8)
            f.write('v %f %f %f %f %f %f\n' % (points[i][0], points[i][1], points[i][2],
                                               color[0], color[1], color[2]))


def validate(test_set, scale_z, obj_file='./prediction', save=True):
    with torch.no_grad():
        unet.eval()
        all_pred = np.array([], dtype=int)
        valLabels = np.array([], dtype=int)
        val_dt = data.kits_dataset(test_set)
        print("val size:", len(val_dt))
        print("batch size:", data.batch_size)
        val_data_loader = torch.utils.data.DataLoader(val_dt, batch_size=data.batch_size,
                                                      collate_fn=data.val_point_transform,
                                                      # collate_fn=data.val_point_transform,
                                                      shuffle=False)
        num_batches = len(val_data_loader)
        for i, batch in enumerate(val_data_loader):
            print(">>>Processing batch: {}/{}".format(i + 1, num_batches))
            if use_cuda:
                batch['x'][1] = batch['x'][1].cuda()
            predictions = unet(batch['x'])
            predictions = predictions.cpu().numpy()
            predictions = np.argmax(predictions, axis=1)
            all_pred = np.concatenate((all_pred, predictions))
            valLabels = np.concatenate((valLabels, batch['y']))

            # save predicted obj
            if save == True:
                xyz = batch['x'][0].cpu().numpy()
                rgb = batch['x'][1].cpu().numpy()
                inds = (xyz[:, 3] == 0)  # get 1st scene in the batch
                xyz = xyz[inds]
                xyz[:, 0] = (xyz[:, 0].astype(np.float32) / scale_z).astype(np.int)
                rgb = rgb[inds]
                pred = predictions[inds]
                out_file = obj_file + str(i) + '.obj'
                save_to_obj(xyz, rgb, pred, out_file)

        class_ious = iou.evaluate(all_pred, valLabels)
        return class_ious


def get_objects(test_set, scale_z, save, folder):
    with torch.no_grad():
        unet.eval()
        all_pred = np.array([], dtype=int)
        valLabels = np.array([], dtype=int)
        val_dt = data.kits_dataset(test_set)
        val_data_loader = torch.utils.data.DataLoader(val_dt, batch_size=1,
                                                      collate_fn=data.val_point_transform,
                                                      shuffle=False)
        num_batches = len(val_data_loader)
        for i, batch in enumerate(val_data_loader):
            print(">>>Processing batch: {}/{}".format(i + 1, num_batches))
            if use_cuda:
                batch['x'][1] = batch['x'][1].cuda()
            predictions = unet(batch['x'])
            predictions = predictions.cpu().numpy()
            predictions = np.argmax(predictions, axis=1)
            valLabels = batch['y'].cpu().numpy()

            # find kidney and save corresponding point cloud
            xyz = batch['x'][0].cpu().numpy()
            rgb = batch['x'][1].cpu().numpy()

            inds = (predictions == 1)  # get kidney class
            inds_gt = inds

            # predicted points
            xyz_pr = xyz[inds]
            xyz_pr[:, 0] = (xyz_pr[:, 0].astype(np.float32) / scale_z).astype(np.int)
            rgb_pr = rgb[inds]
            pred = predictions[inds]

            # gt points
            xyz_gt = xyz[inds_gt]
            xyz_gt[:, 0] = (xyz_gt[:, 0].astype(np.float32) / scale_z).astype(np.int)
            rgb_gt = rgb[inds_gt]
            gt = valLabels[inds_gt]

            if save == True:
                file_name = folder + "/" + batch['case'][:-4] + "-kidney"
                out_pth_file = file_name + ".pth"
                torch.save((xyz_pr, rgb_pr, gt), out_pth_file)

                # save obj
                out_obj_file_pr = file_name + ".obj"
                save_to_obj(xyz_pr, rgb_pr, pred, out_obj_file_pr)
                out_obj_file_gt = file_name + "-gt.obj"
                save_to_obj(xyz_gt, rgb_gt, gt, out_obj_file_gt)
        return


def get_objects_with_coords(test_set, scale_z, save, folder):
    with torch.no_grad():
        unet.eval()
        # all_pred = np.array([], dtype=int)
        # valLabels = np.array([], dtype=int)
        val_dt = data.kits_dataset(test_set)
        val_data_loader = torch.utils.data.DataLoader(val_dt, batch_size=1,
                                                      collate_fn=data.val_point_transform_with_coords,
                                                      shuffle=False)
        num_batches = len(val_data_loader)
        for i, batch in enumerate(val_data_loader):
            print(">>>Processing batch: {}/{}".format(i + 1, num_batches))
            if use_cuda:
                batch['x'][1] = batch['x'][1].cuda()
            predictions = unet(batch['x'])
            predictions = predictions.cpu().numpy()
            predictions = np.argmax(predictions, axis=1)
            valLabels = batch['y'].cpu().numpy()
            coords = batch['coords'].cpu().numpy()

            # find kidney and save corresponding point cloud
            xyz = batch['x'][0].cpu().numpy()
            rgb = batch['x'][1].cpu().numpy()

            inds = (predictions == 1)  # get kidney class
            inds_gt = inds

            # predicted points
            xyz_pr = xyz[inds]
            xyz_pr[:, 0] = (xyz_pr[:, 0].astype(np.float32) / scale_z).astype(np.int)
            rgb_pr = rgb[inds]
            pred = predictions[inds]
            coords = coords[inds]

            # gt points
            xyz_gt = xyz[inds_gt]
            xyz_gt[:, 0] = (xyz_gt[:, 0].astype(np.float32) / scale_z).astype(np.int)
            rgb_gt = rgb[inds_gt]
            gt = valLabels[inds_gt]

            if save == True:
                file_name = folder + "/" + batch['case'][:-4] + "-kidney"
                out_pth_file = file_name + ".pth"
                torch.save((xyz_pr, rgb_pr, gt, coords), out_pth_file)

                # save obj
                out_obj_file_pr = file_name + ".obj"
                save_to_obj(xyz_pr, rgb_pr, pred, out_obj_file_pr)
                out_obj_file_gt = file_name + "-gt.obj"
                save_to_obj(xyz_gt, rgb_gt, gt, out_obj_file_gt)
        return


def get_kidney_prediction(test_set, scale_z, save, folder):
    with torch.no_grad():
        unet.eval()
        val_dt = data.kits_dataset(test_set)
        val_data_loader = torch.utils.data.DataLoader(val_dt, batch_size=1,
                                                      collate_fn=data.val_point_transform_with_coords,
                                                      shuffle=False)
        num_batches = len(val_data_loader)
        for i, batch in enumerate(val_data_loader):
            print(">>>Processing batch: {}/{}".format(i + 1, num_batches))
            if use_cuda:
                batch['x'][1] = batch['x'][1].cuda()
            predictions = unet(batch['x'])
            predictions = predictions.cpu().numpy()
            predictions = np.argmax(predictions, axis=1)
            valLabels = batch['y'].cpu().numpy()
            coords = batch['coords'].cpu().numpy()

            # find kidney and save corresponding point cloud
            xyz = batch['x'][0].cpu().numpy()
            rgb = batch['x'][1].cpu().numpy()
            xyz[:, 0] = (xyz[:, 0].astype(np.float32) / scale_z).astype(np.int)
            gt = valLabels

            if save == True:
                file_name = folder + "/" + batch['case'][:-4] + "-predicted"
                out_pth_file = file_name + ".pth"
                torch.save((xyz, rgb, predictions, coords), out_pth_file)

                # save obj
                out_obj_file_pr = file_name + ".obj"
                save_to_obj(xyz, rgb, predictions, out_obj_file_pr)
                out_obj_file_gt = file_name + "-gt.obj"
                save_to_obj(xyz, rgb, gt, out_obj_file_gt)
        return


def parse_args():
    parser = argparse.ArgumentParser(description='ScanNet validation')
    parser.add_argument('--weights',
                        # default='weights/kidney/detect_kidney.pth', # 1st phase
                        default='weights/2nd_phase/2nd_phase_weights.pth', # 2nd phase
                        help='Model weights')
    parser.add_argument('--test_set', default='./test', help='Folder with samples to validate')
    parser.add_argument('--save', default=False, help='Whether to save obj model')
    parser.add_argument('--obj_file', default='./prediction', help='Output obj filename')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # NOTE: set appropriate dataset flags in data.py
    print('original_full_ds =', data.original_full_ds)
    print('tissue_kidney_full_ds =', data.tissue_kidney_full_ds)
    print('tissue_kidney_tumor_small_ds =', data.tissue_kidney_tumor_small_ds)

    args = parse_args()

    weights = args.weights
    use_cuda = torch.cuda.is_available()
    print("use_cuda =", use_cuda)
    print("weights =", args.weights)
    unet = data.Model(data.dimension, data.full_scale, data.num_cl)
    unet.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    if use_cuda:
        unet = unet.cuda()

    validate(test_set=args.test_set, scale_z=data.scale_z, obj_file=args.obj_file, save=args.save)

    # generate kidney dataset
    # get_objects(test_set="./val", scale_z=data.scale_z, save=True)
    # get_objects_with_coords(test_set="./test", scale_z=data.scale_z, save=True, folder="./kidney_coords_train")
    # get_kidney_prediction(test_set="./kidney_coords_train", scale_z=data.scale_z, save=True, folder="./kidney_coords_train/2nd_phase")

