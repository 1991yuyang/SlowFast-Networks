import torch as t
from torch import nn
from model import SlowFastNet
import os
import json
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


conf = json.load(open("conf.json", "r"))
common_conf = conf["common"]
predict_conf = conf["predict"]
clip_len = common_conf["clip_len"]
slow_tao = common_conf["slow_tao"]
alpha = common_conf["alpha"]
is_group_conv = common_conf["is_group_conv"]
class_names = common_conf["class_names"]
video_pth = predict_conf["video_pth"]
clip_count = predict_conf["clip_count"]
short_side_size = predict_conf["short_side_size"]
crop_size = predict_conf["crop_size"]
batch_size = predict_conf["batch_size"]
use_best_model = predict_conf["use_best_model"]
num_classes = len(class_names)


def load_model():
    model = SlowFastNet(num_classes=num_classes, slow_tao=slow_tao, alpha=alpha, is_group_conv=is_group_conv)
    model = nn.DataParallel(module=model, device_ids=[0])
    if use_best_model:
        model.load_state_dict(t.load("best.pth"))
    else:
        model.load_state_dict(t.load("epoch.pth"))
    model = model.cuda(0)
    model.eval()
    return model


model = load_model()


def load_data():
    cap = cv2.VideoCapture(video_pth)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width < frame_height:
        new_width = short_side_size
        resize_ratio = new_width / frame_width
        new_height = int(resize_ratio * frame_height)
    else:
        new_height = short_side_size
        resize_ratio = new_height / frame_height
        new_width = int(resize_ratio * frame_width)
    buffers = []  # 形状为，[D, H, W, C]
    for _ in range(frame_count):
        _, frame = cap.read()
        # 1.视频帧转rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2.按照短边尺寸进行resize
        frame = cv2.resize(frame, (new_width, new_height))
        buffers.append(frame)
    buffers = np.array(buffers)