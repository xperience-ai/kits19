import nibabel as nib
import numpy as np
import os
import torch
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

color_map = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 0, 255)
]


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`.
    `p`: `NxK` coordinates of `N` points in `K` dimensions.
    `hull` is either a scipy.spatial.Delaunay object
    or the `MxK` array of the coordinates of `M` points in `K`dimensions for which Delaunay triangulation will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def get_convex_hull(points):
    hull = ConvexHull(points)
    v = []
    for vert in hull.vertices:
        v.append(points[vert])
    return v


def pth_to_nii(pth_file, nii_file, original_nii):
    case_path_vol = original_nii
    if not (os.path.exists(case_path_vol)):
        print("ERROR: can't access file", case_path_vol)
        return

    vol_image = nib.load(case_path_vol)
    vol = vol_image.get_data()

    # set seg values to 0 (tissue), then add labels from prediction
    vol.fill(0)

    xyz, rgb, labels, coords = torch.load(pth_file)
    for index in range(len(coords)):
        slice = coords[index][0]
        x = coords[index][1]
        y = coords[index][2]
        label = labels[index]
        vol[slice, x, y] = label

    nib.save(vol_image, nii_file)


# clusters in 2d
def add_convex_hull2d(seg, xyz, labels, show=False, paint_hull=True):
    # paint_hull: paint whole hull or put only points for labels

    # DBSCAN parameters
    min_points_threshold = {"kidney": 30, "tumor": 30}
    eps = {"kidney": 30, "tumor": 30}
    min_samples = {"kidney": 20, "tumor": 20}

    xyzl = np.c_[xyz, labels]  # add columns to the right, xyzl: [slice, x, y, label]
    unique_slices = np.unique(xyzl[:, 0])
    for z in unique_slices:
        slice_pts_indices = xyzl[:, 0] == z
        slice_pts = xyzl[slice_pts_indices, :]
        kidney_points = slice_pts[slice_pts[:, -1] == 1]
        tumor_points = slice_pts[slice_pts[:, -1] == 2]

        if len(kidney_points) > min_points_threshold["kidney"]:
            dbs = DBSCAN(eps=eps["kidney"], min_samples=min_samples["kidney"]).fit(kidney_points[:, 1:3])
            clusters = np.array(dbs.labels_)
            # print("2d kidney clusters:", clusters.max() + 1)
            for c in range(clusters.min(), clusters.max() + 1):
                if c == -1:
                    continue
                ind = clusters == c
                kidney = kidney_points[ind]
                v = get_convex_hull(kidney[:, 1:3])
                v = np.array([v])  # add dimension for cv2.drawContours
                color = 1  # kidney
                img = seg[z].copy()
                img.fill(0)
                cv2.drawContours(image=img, contours=v, contourIdx=-1, color=color, thickness=-1)
                img = img.T
                x = np.nonzero(img)[0]
                y = np.nonzero(img)[1]
                if paint_hull:
                    seg[z, x, y] = color
                else:
                    seg[z, kidney[:, 1], kidney[:, 2]] = 1  # points only

        if len(tumor_points) > min_points_threshold["tumor"]:
            dbs = DBSCAN(eps=eps["tumor"], min_samples=min_samples["tumor"]).fit(tumor_points[:, 1:3])
            clusters = np.array(dbs.labels_)
            # print("2d tumor clusters:", clusters.max() + 1)
            for c in range(clusters.min(), clusters.max() + 1):
                if c == -1:
                    continue
                ind = clusters == c
                tumor = tumor_points[ind]
                v = get_convex_hull(tumor[:, 1:3])
                v = np.array([v])  # add dimension for cv2.drawContours
                color = 2  # tumor
                img = seg[z].copy()
                img.fill(0)
                cv2.drawContours(image=img, contours=v, contourIdx=-1, color=color, thickness=-1)
                img = img.T
                x = np.nonzero(img)[0]
                y = np.nonzero(img)[1]
                if paint_hull:
                    seg[z, x, y] = color
                else:
                    seg[z, tumor[:, 1], tumor[:, 2]] = 2  # points only
    return seg


# clusters in 3d
def add_convex_hull3d(seg, xyz, labels, show=False, paint_hull=True):
    # paint_hull: paint whole hull or put only points for labels

    # DBSCAN parameters
    min_slice_points_threshold = {"kidney": 20, "tumor": 20}
    eps = {"kidney": 20, "tumor": 20}
    min_samples = {"kidney": 1000, "tumor": 100}

    xyzl = np.c_[xyz, labels]  # add columns to the right, xyzl: [slice, x, y, label]

    kidney_points = xyzl[xyzl[:, 3] == 1]
    if len(kidney_points) > min_slice_points_threshold["kidney"]:
        min_samples["kidney"] = len(kidney_points) // 20
        print("kidney points:", len(kidney_points))
        print("kidney min_samples:", min_samples["kidney"])
        kidney_3d_dbs = DBSCAN(eps=eps["kidney"], min_samples=min_samples["kidney"], n_jobs=-1).\
                               fit(kidney_points[:, 0:3])
        kidney_clusters = np.array(kidney_3d_dbs.labels_)
        print("3d kidney clusters:", kidney_clusters.max() + 1)
        kidney_points = np.c_[kidney_points, kidney_clusters] # [slice, x, y, label, cluster]

        kidney_slices = np.unique(kidney_points[:, 0])
        for z in kidney_slices:
            slice_pts_indices = kidney_points[:, 0] == z
            slice_pts = kidney_points[slice_pts_indices, :]

            for c in range(kidney_clusters.min(), kidney_clusters.max() + 1):
                if c == -1:
                    continue
                ind = slice_pts[:, 4] == c
                kidney = slice_pts[ind]

                if len(kidney) > min_slice_points_threshold["kidney"]:
                    v = get_convex_hull(kidney[:, 1:3])
                    v = np.array([v])  # add dimension for cv2.drawContours
                    color = 1  # kidney
                    img = seg[z].copy()
                    img.fill(0)
                    cv2.drawContours(image=img, contours=v, contourIdx=-1, color=color, thickness=-1)
                    img = img.T
                    x = np.nonzero(img)[0]
                    y = np.nonzero(img)[1]
                    if paint_hull:
                        seg[z, x, y] = color
                    else:
                        seg[z, kidney[:, 1], kidney[:, 2]] = color  # points only



    tumor_points = xyzl[xyzl[:, 3] == 2]
    if len(tumor_points) > min_slice_points_threshold["tumor"]:

        min_samples["tumor"] = len(tumor_points) // 20
        print("tumor points:", len(tumor_points))
        print("tumor min_samples:", min_samples["tumor"])
        tumor_3d_dbs = DBSCAN(eps=eps["tumor"], min_samples=min_samples["tumor"]).\
                              fit(tumor_points[:, 0:3])
        tumor_clusters = np.array(tumor_3d_dbs.labels_)
        print("3d tumor clusters:", tumor_clusters.max() + 1)
        tumor_points = np.c_[tumor_points, tumor_clusters]


        tumor_slices = np.unique(tumor_points[:, 0])
        for z in tumor_slices:
            slice_pts_indices = tumor_points[:, 0] == z
            slice_pts = tumor_points[slice_pts_indices, :]

            for c in range(tumor_clusters.min(), tumor_clusters.max() + 1):
                if c == -1:
                    continue
                ind = slice_pts[:, 4] == c
                tumor = slice_pts[ind]

                if len(tumor) > min_slice_points_threshold["tumor"]:
                    v = get_convex_hull(tumor[:, 1:3])
                    v = np.array([v])  # add dimension for cv2.drawContours
                    color = 2  # tumor
                    img = seg[z].copy()
                    img.fill(0)
                    cv2.drawContours(image=img, contours=v, contourIdx=-1, color=color, thickness=-1)
                    img = img.T
                    x = np.nonzero(img)[0]
                    y = np.nonzero(img)[1]
                    if paint_hull:
                        seg[z, x, y] = color
                    else:
                        seg[z, tumor[:, 1], tumor[:, 2]] = color  # points only

    return seg


def pth_samples_to_nii(pth_files, nii_file, nii_file2, original_nii):
    case_path_vol = original_nii
    if not (os.path.exists(case_path_vol)):
        print("ERROR: can't access file", case_path_vol)
        return

    vol_image = nib.load(case_path_vol)
    # vol = vol_image.get_data().astype(np.uint8)
    # vol = vol_image.get_data()
    seg = np.asarray(vol_image.dataobj).astype(np.uint8)
    # seg = vol.astype(np.uint8)

    # set seg values to 0 (tissue), then add labels from prediction
    seg.fill(0)

    for i, pth_file in enumerate(pth_files):
        xyz, rgb, labels, coords = torch.load(pth_file)
        print("Adding convex hull for", pth_file)
        total_cloud_size = len(xyz)
        kidney_cloud_size = len(labels[labels == 1])
        print("point cloud size:", total_cloud_size)
        print("kidney cloud size:", kidney_cloud_size)
        tmp = seg.copy()
        if kidney_cloud_size > 130000: # to fit 32GB RAM
            print("Using convex_hull2d")
            tmp = add_convex_hull2d(tmp, coords, labels)
        else:
            print("Using convex_hull3d")
            tmp = add_convex_hull3d(tmp, coords, labels)
        # tmp = add_convex_hull2(tmp, coords, labels)
        # process seg images separately
        for z in range(seg.shape[0]):
            x = np.nonzero(tmp[z])[0]
            y = np.nonzero(tmp[z])[1]
            seg[z, x, y] = tmp[z, x, y]


    new_image = nib.Nifti1Image(seg, vol_image.affine, vol_image.header)
    nib.save(new_image, nii_file)
    nib.save(new_image, nii_file2)


def nii_to_pth_no_labels(nii_file, out_file="imaging.pth", keep_percentage=0.03,
                         hu_min=-512, hu_max=512):
    '''
    Converts nii format to pth: 3 arrays [[x,y,z]], [intensity], [label]
    :param nii_file: input nii file
    :param out_file: output .pth filename
    :param hu_min: min intensity value to clip
    :param hu_max: max intensity value to clip
    :param keep_percentage: export only <keep_percentage> of points with max intensity
    '''

    # load segmentation and volume (from specific case_##### folder in the kits19 dataset)
    print("Converting {} to {}".format(nii_file, out_file))
    case_path_vol = nii_file

    if not (os.path.exists(case_path_vol)):
        print("ERROR: can't access file", case_path_vol)
        return

    vol = nib.load(case_path_vol)
    spacing = vol.affine
    spc_ratio = np.abs(np.sum(spacing[2, :])) / np.abs(np.sum(spacing[0, :]))
    vol = vol.get_data()

    # clip color intensity at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(vol, hu_min, hu_max)
    mxval = np.max(volume)
    mnval = np.min(volume)

    # re-scale values to [-1..1] interval
    vol_ims = 2 * (volume - mnval) / max(mxval - mnval, 1e-3) - 1
    threshold = np.percentile(vol_ims, 100 * (1 - keep_percentage))

    vol_xyz = []  # [[z,x,y],...]
    gray = []  # grayscale, color values are in the range [-1..1]
    labels = []  # 0 - background (tissue), 1 - kidney, 2 - tumor
    coords = []  # [[slice,x,y],...] - to keep mapping between original nii and sub cloud points

    total_points = 0
    saved_points = 0
    for i in tqdm(range(vol_ims.shape[0]), unit="slices"):
        for x in range(0, vol_ims.shape[1]):
            for y in range(0, vol_ims.shape[2]):
                total_points += 1
                color = vol_ims[i][x][y]
                if color > threshold:
                    saved_points += 1
                    vol_xyz.append([i * spc_ratio, x, y])
                    gray.append([color, color, color])  # 3 channels (identical)
                    # gray.append(color) # 1 channel
                    labels.append(0)
                    # labels.append(seg[i][x][y])
                    coords.append([i, x, y])

    vol_xyz = np.array(vol_xyz).astype(np.float32)
    gray = np.array(gray).astype(np.float32)
    labels = np.array(labels).astype(np.int8)
    coords = np.array(coords).astype(np.int16)
    print("Exported {}/{} points ({:.1f}%)".format(saved_points, total_points,
                                                   saved_points / total_points * 100))
    print("Writing to:", out_file)
    torch.save((vol_xyz, gray, labels, coords), out_file)
    return out_file


def nii_to_pth(data_folder, out_file="imaging.pth", keep_percentage=0.03,
               hu_min=-512, hu_max=512):
    '''
    Converts nii format to pth: 3 arrays [[x,y,z]], [intensity], [label]
    :param data_folder: folder with cases
    :param destination_folder: folder to store .pth file
    :param out_filename: output .pth filename
    :param hu_min: min intensity value to clip
    :param hu_max: max intensity value to clip
    :param keep_percentage: export only <keep_percentage> of points with max intensity
    '''

    # load segmentation and volume (from specific case_##### folder in the kits19 dataset)
    print("Converting {} to {}".format(data_folder, out_file))
    case_path_vol = os.path.join(data_folder, "imaging.nii.gz")
    case_path_seg = os.path.join(data_folder, "segmentation.nii.gz")

    if not (os.path.exists(case_path_vol)):
        print("ERROR: can't access file", case_path_vol)
        return
    if not (os.path.exists(case_path_seg)):
        print("ERROR: can't access file", case_path_seg)
        return

    vol = nib.load(case_path_vol)
    seg = nib.load(case_path_seg)
    spacing = vol.affine
    spc_ratio = np.abs(np.sum(spacing[2, :])) / np.abs(np.sum(spacing[0, :]))
    vol = vol.get_data()
    seg = seg.get_data()
    seg = seg.astype(np.int8)  # 0 - background (tissue), 1 - kidney, 2 - tumor

    # clip color intensity at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(vol, hu_min, hu_max)
    mxval = np.max(volume)
    mnval = np.min(volume)

    # re-scale values to [-1..1] interval
    vol_ims = 2 * (volume - mnval) / max(mxval - mnval, 1e-3) - 1
    threshold = np.percentile(vol_ims, 100 * (1 - keep_percentage))

    vol_xyz = []  # [[z,x,y],...]
    gray = []  # grayscale, color values are in the range [-1..1]
    labels = []  # 0 - background (tissue), 1 - kidney, 2 - tumor
    coords = []  # [[slice,x,y],...] - to keep mapping between original nii and sub cloud points

    total_points = 0
    saved_points = 0
    for i in range(vol_ims.shape[0]):
        print("slice: {}/{}".format(i + 1, vol_ims.shape[0]))
        for x in range(0, vol_ims.shape[1]):
            for y in range(0, vol_ims.shape[2]):
                total_points += 1
                color = vol_ims[i][x][y]
                if color > threshold:
                    saved_points += 1
                    vol_xyz.append([i * spc_ratio, x, y])
                    gray.append([color, color, color])  # 3 channels (identical)
                    # gray.append(color) # 1 channel
                    labels.append(seg[i][x][y])
                    coords.append([i, x, y])

    vol_xyz = np.array(vol_xyz).astype(np.float32)
    gray = np.array(gray).astype(np.float32)
    labels = np.array(labels).astype(np.int8)
    coords = np.array(coords).astype(np.int16)
    print("Exported {}/{} points ({:.1f}%)".format(saved_points, total_points,
                                                   saved_points / total_points * 100))
    print("Writing to:", out_file)
    torch.save((vol_xyz, gray, labels, coords), out_file)
    print("Done\n")
    return out_file


def nii_to_pth_split_no_labels(nii_file, out_file="imaging.pth", keep_percentage=0.03, num_samples=3,
                               hu_min=-512, hu_max=512):
    '''
    Converts nii format to pth: 3 arrays [[x,y,z]], [intensity], [label]
    :param nii_file: folder with cases
    :param destination_folder: folder to store .pth file
    :param out_filename: output .pth filename
    :param hu_min: min intensity value to clip
    :param hu_max: max intensity value to clip
    :param keep_percentage: export only <keep_percentage> of points with max intensity
    '''

    # load segmentation and volume (from specific case_##### folder in the kits19 dataset)
    slices = int(1 / keep_percentage)
    if num_samples is None:
        num_samples = slices
    print(
        "Splitting {} to {} slices with {}% points".format(nii_file, slices, keep_percentage * 100, out_file))
    print("Saving first {} samples with top intensity".format(num_samples))

    if not (os.path.exists(nii_file)):
        print("ERROR: can't access file", nii_file)
        return

    vol = nib.load(nii_file)
    spacing = vol.affine
    spc_ratio = np.abs(np.sum(spacing[2, :])) / np.abs(np.sum(spacing[0, :]))
    vol = vol.get_data()

    # clip color intensity at max and min values
    if hu_min is not None or hu_max is not None:
        volume = np.clip(vol, hu_min, hu_max)
    mxval = np.max(volume)
    mnval = np.min(volume)

    # re-scale values to [-1..1] interval
    vol_ims = 2 * (volume - mnval) / max(mxval - mnval, 1e-3) - 1

    # calculate percentile intervals (p) with the width of keep_percentage
    k = keep_percentage
    t = (1 - k) * 100
    step = 1
    p = [1]  # max value
    while t > 0:
        p.append(np.percentile(vol_ims, t))
        step += 1
        t = (1 - k * step) * 100
    p.append(-1)  # min value
    scan_pth_names = []

    for s in range(len(p) - 1):
        if s == num_samples:
            break  # take first num_samples only
        pr_high = p[s]
        pr_low = p[s + 1]

        vol_xyz = []  # [[z,x,y],...]
        gray = []  # grayscale, color values are in the range [-1..1]
        labels = []  # 0 - background (tissue), 1 - kidney, 2 - tumor
        coords = []  # [[slice,x,y],...] - to keep mapping between original nii and sub cloud points

        total_points = 0
        saved_points = 0
        for i in tqdm(range(vol_ims.shape[0]), unit="slices"):
            # for i in range(vol_ims.shape[0]):
            #     print("slice: {}/{}".format(i + 1, vol_ims.shape[0]))
            for x in range(0, vol_ims.shape[1]):
                for y in range(0, vol_ims.shape[2]):
                    total_points += 1
                    color = vol_ims[i][x][y]

                    if color > pr_low and color < pr_high:
                        saved_points += 1
                        vol_xyz.append([i * spc_ratio, x, y])
                        gray.append([color, color, color])  # 3 channels (identical)
                        # gray.append(color) # 1 channel
                        labels.append(0)
                        # labels.append(seg[i][x][y])
                        coords.append([i, x, y])

        vol_xyz = np.array(vol_xyz).astype(np.float32)
        gray = np.array(gray).astype(np.float32)
        labels = np.array(labels).astype(np.int8)
        coords = np.array(coords).astype(np.int16)
        file_name = "{}_{:02d}.pth".format(out_file, s)
        if len(vol_xyz > 0):
            print("Exported {}/{} points ({:.1f}%)".format(saved_points, total_points,
                                                           saved_points / total_points * 100))
            print("Writing to:", file_name)
            torch.save((vol_xyz, gray, labels, coords), file_name)
            scan_pth_names.append(file_name)

    return scan_pth_names


def pth_to_obj(in_file, out_file, color_map=color_map):
    print("Converting {} to {}".format(in_file, out_file))
    data, colors, labels, coords = torch.load(in_file)
    with open(out_file, 'w') as f:
        for i in range(data.shape[0]):
            label = labels[i]
            if label > 0:
                color = color_map[label]
            else:
                color = np.array([colors[i][0], colors[i][1], colors[i][2]])
                if color.max() < 2:  # [-1..1] -> [0..255]
                    color = color.astype(np.float32)
                    color = ((color + 1) / 2 * 255).astype(np.int8)
            f.write(
                'v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


if __name__ == '__main__':
    # for f in sorted(os.walk('./data/datasets/kits19/data/')):
    #     folder = f[0]
    #     _, case = os.path.split(f[0])
    #     case_id = case[-5:]
    #     out_file = "./pth/rgb/imaging_" + case_id + "_3.pth"
    #     nii_to_pth(data_folder=folder, out_file=out_file, keep_percentage=0.05)

    # pth_to_obj(in_file="./scn/SparseConvNet/examples/ScanNet/full_set/imaging_00007_3.pth", out_file="./obj/imaging_00007.obj")
    # pth_to_obj(in_file="./scn/SparseConvNet/examples/ScanNet/full_set/imaging_00053_3.pth", out_file="./obj/imaging_00053.obj")
    # pth_to_obj(in_file="./scn/SparseConvNet/examples/ScanNet/full_set/imaging_00134_3.pth", out_file="./obj/imaging_00134.obj")

    nii_to_pth_no_labels(nii_file="./data/datasets/kits19/data/case_00216/imaging.nii.gz",
                         out_file="./imaging_00216.pth", keep_percentage=0.05)
    pth_to_obj(in_file="./imaging_00216.pth", out_file="./imaging_00216.obj")

    # pth_to_nii(pth_file="./kidney_coords_train/2nd_phase/imaging_00184_3_c-kidney-predicted.pth",
    #            nii_file="./kidney_coords_train/2nd_phase/imaging_00184_3_c-kidney-predicted.nii.gz" ,
    #            original_nii="./data/datasets/kits19/data/case_00184")

    # split nii to num_samples with top intensity, each sample takes keep_percentage of data
    # nii_to_pth_split_no_labels(nii_file="./data/datasets/kits19/data/case_00010/imaging.nii.gz",
    #            out_file="./test/imaging_00010", keep_percentage=0.03, num_samples=3)

    # for f in sorted(glob.glob("./test/*.pth")):
    #     out = f[:-4] + ".obj"
    #     pth_to_obj(f, out)

    pass
