import argparse
import convert
import data
import validate
import numpy as np
import torch
import glob
import os
import torch.utils.data
import time
import visualize


def val_point_transform_with_coords(tbl):
    locs = []
    feats = []
    labels = []
    coords = []
    idx = 0
    a, b, c, d = tbl

    if a.shape[1] > 3:  # temp fix for small dataset, remove 4th column
        a = a[:, 0:3]
    m = np.eye(3)
    m *= data.scale
    m[0][0] *= data.scale_z / data.scale  # scale z axis by scale_z
    a = np.matmul(a, m)
    m = a.min(0)
    M = a.max(0)
    offset = -m + np.clip(data.full_scale - M + m - 0.001, 0, None) + \
             np.clip(data.full_scale - M + m + 0.001, None, 0)
    a += offset

    idxs = (a.min(1) >= 0) * (a.max(1) < data.full_scale)
    a = a[idxs]
    b = b[idxs]
    c = c[idxs]
    d = d[idxs]

    # set max_points to fit to memory
    # if a.shape[0] > data.max_points:
    #     idxs = np.random.choice(a.shape[0], data.max_points, replace=False)
    #     a = a[idxs]
    #     b = b[idxs]
    #     c = c[idxs]
    #     d = d[idxs]

    a = torch.from_numpy(a).long()
    locs.append(torch.cat([a, torch.LongTensor(a.shape[0], 1).fill_(idx)], 1))
    feats.append(torch.from_numpy(b))
    labels.append(torch.from_numpy(c))
    coords.append(torch.from_numpy(d))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    coords = torch.cat(coords, 0)
    return {'x': [locs, feats], 'y': labels.long(), 'id': tbl, 'coords': coords}


def inference_for_samples(test_set, weights1, weights2, save_pngs=False):
    # segmentation.nii.gz and prediction_#####.nii.gz will be put into corresponding case folder
    nii_files = []
    for case in sorted(glob.glob(test_set + "/*/imaging.nii.gz", recursive=True)) \
            :
        t = {"folder": os.path.dirname(case), "file": case}
        nii_files.append(t)
    print("Files in test set:", len(nii_files))

    use_cuda = torch.cuda.is_available()
    print("use_cuda =", use_cuda)
    print("weights1 =", weights1)
    print("weights2 =", weights2)

    unet2 = data.Model(data.dimension, data.full_scale, num_classes=2)
    unet2.load_state_dict(torch.load(weights1, map_location=torch.device('cpu')))
    unet3 = data.Model(data.dimension, data.full_scale, num_classes=3)
    unet3.load_state_dict(torch.load(weights2, map_location=torch.device('cpu')))

    if use_cuda:
        unet2 = unet2.cuda()
        unet3 = unet3.cuda()

    for case in nii_files:
        print("\nProcessing", case["file"])
        # split cloud into num_samples parts and run inference for each (9% of top intensity is enough)
        num_samples = 2
        scan_pth_names = convert.nii_to_pth_split_no_labels(case["file"],
                                                            out_file=case["folder"] + "/imaging",
                                                            keep_percentage=0.03, num_samples=num_samples,
                                                            hu_min=-512,
                                                            hu_max=512)
        print("Point cloud data in:", scan_pth_names)
        with torch.no_grad():
            unet2.eval()
            unet3.eval()

            predicted_pth_names = []
            for i, scan_pth in enumerate(scan_pth_names):
                # phase 1 - detect kidney
                print("\n>>> PHASE 1: detect kidney")

                batch = val_point_transform_with_coords(torch.load(scan_pth))
                if use_cuda:
                    batch['x'][1] = batch['x'][1].cuda()

                predictions = unet2(batch['x'])
                predictions = predictions.cpu().numpy()
                predictions = np.argmax(predictions, axis=1)
                coords = batch['coords'].cpu().numpy()

                # find kidney and save corresponding point cloud
                xyz = batch['x'][0].cpu().numpy()
                rgb = batch['x'][1].cpu().numpy()
                inds = (predictions == 1)  # get kidney class

                # predicted points
                xyz_pr = xyz[inds]
                xyz_pr[:, 0] = (xyz_pr[:, 0].astype(np.float32) / data.scale_z).astype(np.int)
                rgb_pr = rgb[inds]
                pred = predictions[inds]
                coords = coords[inds]

                file_name = "{}/imaging_kidney_{:02d}".format(case["folder"], i)
                out_pth_file = file_name + ".pth"
                torch.save((xyz_pr, rgb_pr, pred, coords), out_pth_file)
                # save obj
                out_obj_file_pr = file_name + ".obj"
                validate.save_to_obj(xyz_pr, rgb_pr, pred, out_obj_file_pr)

                print("\n>>> PHASE 2: detect tissue, kidney and tumor")
                batch = val_point_transform_with_coords(torch.load(out_pth_file))
                if use_cuda:
                    batch['x'][1] = batch['x'][1].cuda()
                predictions = unet3(batch['x'])
                predictions = predictions.cpu().numpy()
                predictions = np.argmax(predictions, axis=1)
                coords = batch['coords'].cpu().numpy()

                # find kidney and save corresponding point cloud
                xyz = batch['x'][0].cpu().numpy()
                rgb = batch['x'][1].cpu().numpy()
                xyz[:, 0] = (xyz[:, 0].astype(np.float32) / data.scale_z).astype(np.int)

                # file_name = case["folder"] + "/imaging-kidney-predicted"
                file_name = "{}/imaging_kidney_predicted_{:02d}".format(case["folder"], i)
                out_pth_file = file_name + ".pth"
                torch.save((xyz, rgb, predictions, coords), out_pth_file)
                predicted_pth_names.append(out_pth_file)

                # save obj
                out_obj_file_pr = file_name + ".obj"
                validate.save_to_obj(xyz, rgb, predictions, out_obj_file_pr)

            # save final segmentation.nii.gz
            case_number = case["folder"][-5:]
            convert.pth_samples_to_nii(pth_files=predicted_pth_names,
                                       nii_file=case["folder"] + "/segmentation.nii.gz",
                                       nii_file2=case["folder"] + "/prediction_" + case_number + ".nii.gz",
                                       original_nii=case["file"])
            print("Saved segmentation to:", case["folder"] + "/segmentation.nii.gz")

            file_size = os.path.getsize(case["folder"] + "/imaging.nii.gz")  # in bytes

            if save_pngs and file_size < 100000000:  # ~100Mb, Memory error for big scans
                print("Saving png files")
                visualize.visualize(
                    path=case["folder"],
                    destination=case["folder"] + "/png",
                    hu_min=-512,
                    hu_max=512,
                    plane="axial"
                )
            print("*" * 70)


if __name__ == '__main__':
    weights1 = './weights/kidney/detect_kidney.pth'  # 1st phase
    weights2 = './weights/2nd_phase/2nd_phase_weights.pth'  # 2nd phase

    inference_for_samples(test_set="./test", weights1=weights1, weights2=weights2, save_pngs=True)
