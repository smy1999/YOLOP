import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.lane_root = lane_root / indicator
        self.lane_list = self.lane_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lane_label = cv2.imread(data["lane"], 0)
        # print(lane_label.shape)
        # print(seg_label.shape)
        # print(lane_label.shape)
        # print(seg_label.shape)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        
        (img, lane_label), ratio, pad = letterbox((img, lane_label), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
            
        if self.is_train:
            combination = (img, lane_label)
            (img, lane_label) = random_perspective(
                combination=combination,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            #print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                lane_label = np.fliplr(lane_label)

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                lane_label = np.filpud(lane_label)

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)

        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
            
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)

        target = [lane_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_lane = []
        for i, l in enumerate(label):
            label_lane.append(l[0])
        return torch.stack(img, 0), [torch.stack(label_lane, 0)], paths, shapes

