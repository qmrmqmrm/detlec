import os, sys
import zipfile
import tarfile
from PIL import Image
import glob
import os.path as op
import numpy as np
import json
import cv2
import math
import torch
from torch.utils.data import Dataset, DataLoader

from my_torch.loader.loader_basic import Dataset_Base
from my_torch.loader import get_calibration_form as gc


class A2D2_Loader(Dataset_Base):
    def __init__(self, path):
        self.calib_path = path
        calib_dict = gc.get_calibration(self.calib_path)

        self.camera_path = os.path.join(self.calib_path, 'image')
        self.label_path = self.camera_path.replace("/image", "/label")
        self.ann = self.get_anno_data(calib_dict)
        super().__init__(self.ann)

    def get_anno_data(self, calib_dict):
        label_files = sorted(glob.glob(os.path.join(self.label_path, '*.json')))
        img_files = sorted(glob.glob(os.path.join(self.camera_path, '*.png')))

        images = list()

        num_image = len(img_files)
        count = 0
        for i, (img_file, lable_file) in enumerate(zip(img_files, label_files)):
            img = cv2.imread(img_file)
            image = dict()

            with open(lable_file, 'r') as f:
                label = json.load(f)
                # ['2d_bbox', '3d_points', 'alpha', 'axis', 'center', 'class', 'id', 'occlusion', 'rot_angle', 'size', 'truncation']
            anns = self.convert_bev(label, img, calib_dict, viewpoint=True, vp_res=True, bins=12)

            if len(anns) != 0:
                image['width'], image['height'] = img.shape[:2]
                image['file_name'] = img_file
                image["id"] = count
                image["annotations"] = anns
                images.append(image)
                count += 1
            print_progress(f"{i} /{num_image}")
        return images

    def convert_bev(self, label, img, calib, vp_res, bins, bvres=0.05, viewpoint=False):
        """


        """
        ann_id = 0
        annotations = list()
        for boxes, obj in label.items():
            location = np.array(obj['center']).reshape((1, 3))
            pts_3d_ref = np.transpose(np.dot(np.linalg.inv(calib["R0"]), np.transpose(location)))
            n = pts_3d_ref.shape[0]
            pts_3d_hom = np.hstack((pts_3d_ref, np.ones((n, 1))))
            pv = np.dot(pts_3d_hom, np.transpose(calib["C2V"]))
            categories = ['Car', 'Pedestrian', 'Cyclist']
            category_dict = {k: v for v, k in enumerate(categories)}

            o = obj['class']
            label = category_dict.get(o, 8)  # Default value just in case

            if (label != 7) and (label != 8):

                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, x, y = obtain_bvbox(obj, img, pv, 0.05)

                if bbox_xmin < 0:
                    continue

                ann = {}
                ann['ann_id'] = ann_id
                ann_id += 1
                ann['category_id'] = label

                ann['bbox'] = [bbox_xmin, bbox_ymin, np.abs(bbox_xmax - bbox_xmin), np.abs(bbox_ymax - bbox_ymin)]
                #     # ONLY VALID FOR FRONTAL CAMERA (ONLY_FRONT PARAM)
                velodyne_h = 1.12
                ann['height'] = [obj['size'][0] * 255 / 3.0, ((pv[0][2] + velodyne_h) + obj['size'][
                    0] * 0.5) * 255 / 3.0]  # (p[0][2]+velodyne_h)]#Not codificated ground
                ann['bbox3D'] = [(bbox_xmin + bbox_xmax) / 2., (bbox_ymin + bbox_ymax) / 2.,
                                 round(obj['size'][1] / bvres, 3), round(obj['size'][2] / bvres, 3)]
                ann['segmentation'] = [
                    [bbox_xmin, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax, bbox_ymax, bbox_xmax, bbox_ymin]]
                ann['area'] = math.fabs(bbox_xmax - bbox_xmin) * math.fabs(bbox_ymax - bbox_ymin)
                ann['iscrowd'] = 0
                if viewpoint:
                    ann['viewpoint'] = [rad2bin(obj['rot_angle'], bins), obj['rot_angle']] if vp_res else [
                        rad2bin(obj['rotation_y'], bins)]
                annotations.append(ann)
        return annotations


def rad2bin(rad, bins):
    bin_dist = np.linspace(-math.pi, math.pi, bins + 1)  # for each class (bins*n_classes)
    bin_res = (bin_dist[1] - bin_dist[0]) / 2.
    bin_dist = [bin - bin_res for bin in
                bin_dist]  # Substracting half of the resolution to each bin it obtains one bin for each direction (N,W,S,E)
    for i_bin in range(len(bin_dist) - 1):
        if bin_dist[i_bin] <= rad and bin_dist[i_bin + 1] >= rad:
            return i_bin

    return 0  # If the angle is above max angle, it won't match so it corresponds to initial bin, initial bin must be from (-pi+bin_res) to (pi-bin_res)


def obtain_bvbox(obj, bv_img, pv, bvres=0.05):
    bvrows, bvcols, _ = bv_img.shape
    centroid = [round(num, 2) for num in pv[0][:2]]  # Lidar coordinates
    #
    length = obj['size'][0]
    width = obj['size'][1]
    yaw = obj['rot_angle']

    # # Compute the four vertexes coordinates
    corners = np.array([[centroid[0] - length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] + width / 2.],
                        [centroid[0] + length / 2., centroid[1] - width / 2.],
                        [centroid[0] - length / 2., centroid[1] - width / 2.]])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    rotated_corners = np.dot(corners - centroid, R) + centroid

    x1 = bvcols / 2 + min(rotated_corners[:, 0]) / bvres
    x2 = bvcols / 2 + max(rotated_corners[:, 0]) / bvres
    y1 = bvrows - max(rotated_corners[:, 1]) / bvres
    y2 = bvrows - min(rotated_corners[:, 1]) / bvres

    x = bvcols / 2 + (rotated_corners[:, 0]) / bvres
    y = bvrows - (rotated_corners[:, 1]) / bvres

    roi = bv_img[int(y1):int(y2), int(x1):int(x2)]
    nonzero = np.count_nonzero(np.sum(roi, axis=2))
    if nonzero < 3:  # Detection is doomed impossible with fewer than 3 points
        return -1, -1, -1, -1, None, None
    # # TODO: Assign DontCare labels to objects with few points?
    #
    # # Remove objects outside the BEV image
    if x1 <= 0 and x2 <= 0 or \
            x1 >= bvcols - 1 and x2 >= bvcols - 1 or \
            y1 <= 0 and y2 <= 0 or \
            y1 >= bvrows - 1 and y2 >= bvrows - 1:
        return -1, -1, -1, -1, None, None  # Out of bounds
    #
    # # Clip boxes to the BEV image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bvcols - 1, x2)
    y2 = min(bvrows - 1, y2)

    return x1, y1, x2, y2, x, y


def load_data():
    path = "/media/dolphin/intHDD/a2d2"
    data_load = A2D2_Loader(path)

    train_loader = DataLoader(dataset=data_load,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2)


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


if __name__ == '__main__':
    load_data()
