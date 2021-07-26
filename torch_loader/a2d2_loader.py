import os, sys
import zipfile
import tarfile
from PIL import Image
import glob
import os.path as op
import numpy as np
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils.util_function import print_progress
from loader_basic import Dataset_Base
import get_calibration_form as gc


class A2D2_Loader(Dataset_Base):
    def __init__(self, path):
        self.path = path
        self.calib_path = path
        self.calib_dict = gc.get_calibration(self.calib_path)
        self.camera_path = os.path.join(path,
                                        'bbox/camera_lidar_semantic_bboxes/*/camera/cam_front_center/')
        self.label_path = self.camera_path.replace("/camera/", "/label3D/")
        self.img_files, self.imgs = self.get_imge_data()
        self.ann = self.get_anno_data()
        super().__init__(self.imgs, self.ann)

    def get_imge_data(self):
        img_files = sorted(glob.glob(os.path.join(self.camera_path, '*.png')))

        imgs = dict()
        num_files = len(img_files)
        for i, img_file in enumerate(img_files):
            img = cv2.imread(img_file)
            img_np = np.array(img)
            imgs[img_file] = img_np
            # print_progress(f"-- Progress: {i}/{num_files}")
        return img_files, imgs

    def get_anno_data(self):
        label_files = sorted(glob.glob(os.path.join(self.label_path, '*.json')))
        label_dict = dict()
        for i, file_name in enumerate(label_files):
            with open(file_name, 'r') as f:
                label = json.load(f)
            label_dict[file_name] = label
        print(label_dict)
        return label_dict


def load_data():
    path = "/media/dolphin/intHDD/a2d2"
    data_load = A2D2_Loader(path)

    train_loader = DataLoader(dataset=data_load,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)
    print(len(train_loader))


def print_progress(status_msg):
    # NOTE: the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


if __name__ == '__main__':
    load_data()
