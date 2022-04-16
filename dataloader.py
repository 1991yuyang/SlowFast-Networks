from torch.utils import data
import cv2
import os
import torch as t
from numpy import random as rd
import numpy as np
"""
data_root:
    class1
        1.avi
        2.avi
        ...
    class2
        1.avi
        2.avi
        ...
    ....
"""


class MySet(data.Dataset):

    def __init__(self, crop_size, short_side_size_range, clip_len, data_root, is_train, class_names):
        """

        :param crop_size: 视频帧裁剪尺寸[height, width]，例如[224, 224]
        :param short_side_size_range: 视频帧resize短边尺寸范围，例如[256, 320]
        :param clip_len: 提取视频帧帧数,至少为fastpath的采样步长
        :param data_root: 数据根目录，ucf101格式
        :param is_train: True表示训练集，False表示验证集
        :param class_names: 类别名称列表
        """
        self.class_names = class_names
        self.crop_size = crop_size
        self.short_side_size_range = short_side_size_range
        self.clip_len = clip_len
        self.data_root = data_root
        self.is_train = is_train
        self.sub_dir_names = os.listdir(self.data_root)
        class_video_pths = [[[sub_dir, os.path.join(self.data_root, sub_dir, img_name)] for img_name in os.listdir(os.path.join(self.data_root, sub_dir))] for sub_dir in self.sub_dir_names]
        self.class_video_pths = []  # [[sample1_class, sample1_video_pth], [sample2_class, sample2_video_pth], ....]
        for i in class_video_pths:
            for j in i:
                self.class_video_pths.append(j)

    def __getitem__(self, index):
        class_name, video_pth = self.class_video_pths[index]
        l = t.tensor(self.class_names.index(class_name)).type(t.LongTensor)
        d = self.load_one_video(video_pth)
        return d, l

    def __len__(self):
        return len(self.class_video_pths)

    def load_one_video(self, video_pth):
        cap = cv2.VideoCapture(video_pth)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        short_side_size = rd.randint(*self.short_side_size_range)
        if frame_width < frame_height:
            new_width = short_side_size
            resize_ratio = new_width / frame_width
            new_height = int(resize_ratio * frame_height)
        else:
            new_height = short_side_size
            resize_ratio = new_height / frame_height
            new_width = int(resize_ratio * frame_width)
        buffers = []  # 形状为，[D, H, W, C]
        is_flip = rd.random() < 0.5
        for _ in range(frame_count):
            _, frame = cap.read()
            # 1.视频帧转rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2.按照短边尺寸进行resize
            frame = cv2.resize(frame, (new_width, new_height))
            # 3.如果是训练则随机进行翻转
            if self.is_train and is_flip:
                frame = cv2.flip(frame, 1)
            buffers.append(frame)
        buffers = np.array(buffers)
        # 4.图像裁剪
        crop_h_start = rd.randint(new_height - self.crop_size[0])
        crop_w_start = rd.randint(new_width - self.crop_size[1])
        buffers = buffers[:, crop_h_start:crop_h_start + self.crop_size[0], crop_w_start:crop_w_start + self.crop_size[1], :]
        # 5.图像归一化
        buffers = buffers / 255
        # 6.时间维度裁切
        buffers = self.tempral_clip(buffers)
        # 7.转为tensor
        d = self.cvt_buffer_to_tensor(buffers)
        return d

    def tempral_clip(self, buffers):
        if buffers.shape[0] < self.clip_len:
            repeat_times = int(np.ceil(self.clip_len / buffers.shape[0]))
            buffers = np.concatenate([buffers] * repeat_times, axis=0)
        clip_start = rd.randint(buffers.shape[0] - self.clip_len)
        buffers = buffers[clip_start:clip_start + self.clip_len, :, :, :]
        return buffers

    def cvt_buffer_to_tensor(self, buffers):
        buffers = np.transpose(buffers, axes=[3, 0, 1, 2])
        d = t.from_numpy(buffers).type(t.FloatTensor)
        return d


def make_loader(crop_size, short_side_size_range, clip_len, data_root, is_train, class_names, batch_size, num_workers):
    loader = iter(data.DataLoader(MySet(crop_size, short_side_size_range, clip_len, data_root, is_train, class_names), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    loader = make_loader((224, 224), (256, 320), 16, r"/home/yuyang/data/ucf101", True, ["ApplyEyeMakeup", "ApplyLipstick", "Archery"], 2, 0)
    # ds = MySet((224, 224), (256, 320), 16, r"/home/yuyang/data/ucf101", True, ["ApplyEyeMakeup", "ApplyLipstick", "Archery"])
    for d, l in loader:
        print(d.size())
        print(l)