from torch.utils import data
import cv2
import os
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

    def __init__(self, crop_size, short_side_size_range, clip_len, data_root, is_train):
        self.crop_size = crop_size
        self.short_side_size_range = short_side_size_range
        self.clip_len = clip_len
        self.data_root = data_root
        self.sub_dir_names = os.listdir(self.data_root)
        class_video_pths = [[[sub_dir, os.path.join(self.data_root, sub_dir, img_name)] for img_name in os.listdir(os.path.join(self.data_root, sub_dir))] for sub_dir in self.sub_dir_names]
        self.class_video_pths = []  # [[sample1_class, sample1_video_pth], [sample2_class, sample2_video_pth], ....]
        for i in class_video_pths:
            for j in i:
                self.class_video_pths.append(j)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    ds = MySet((224, 224), (224, 256), 16, r"/media/yuyang/Seagate Basic/ucf101/UCF-101", True)
