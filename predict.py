import torch as t
from torch import nn
from model import SlowFastNet
import os
import json
import cv2
import numpy as np
from numpy import random as rd
from dataloader import make_predict_loader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


conf = json.load(open("conf.json", "r"))
common_conf = conf["common"]
predict_conf = conf["predict"]
clip_len = common_conf["clip_len"]
slow_tao = common_conf["slow_tao"]
alpha = common_conf["alpha"]
is_group_conv = common_conf["is_group_conv"]
class_names = common_conf["class_names"]
num_workers = common_conf["num_workers"]
video_pth = predict_conf["video_pth"]
clip_count = predict_conf["clip_count"]
short_side_size = predict_conf["short_side_size"]
crop_size = predict_conf["crop_size"]
batch_size = predict_conf["batch_size"]
use_best_model = predict_conf["use_best_model"]
crop_times = predict_conf["crop_times"]
show_video = predict_conf["show_video"]
num_classes = len(class_names)
softmax_op = nn.Softmax(dim=1)


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


def tempral_clip(buffers):
    if buffers.shape[0] < clip_len:
        repeat_times = int(np.ceil(clip_len / buffers.shape[0]))
        buffers = np.concatenate([buffers] * repeat_times, axis=0)
    clip_start = rd.randint(buffers.shape[0] - clip_len)
    buffers = buffers[clip_start:clip_start + clip_len, :, :, :]
    return buffers


def load_data(video):
    """
    param video: 可以填入视频路径，字符串格式；也可以填入帧流，ndarray格式，形状为[D, H, W, C]
    """
    clips = []
    buffers = []  # 形状为，[D, H, W, C]
    if isinstance(video, str):
        # video为video路径
        cap = cv2.VideoCapture(video)
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
        assert np.all(np.array(crop_size) <= np.array((new_height, new_width))), "crop_size > resized_image_size, (%d,%d) > (%d,%d)" % (crop_size[0], crop_size[1], new_height, new_width)
        for _ in range(frame_count):
            _, frame = cap.read()
            if show_video and (cv2.waitKey(35) & 0xff != ord('q')):
                cv2.imshow("frame", frame)
            # 1.视频帧转rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2.按照短边尺寸进行resize
            frame = cv2.resize(frame, (new_width, new_height))
            buffers.append(frame)
    if isinstance(video, np.ndarray):
        # video为视频帧，形状为[D, H, W, C]，D为帧数目，H为高度，W为宽度，C为通道数
        frame_width = int(video.shape[2])
        frame_height = int(video.shape[1])
        if frame_width < frame_height:
            new_width = short_side_size
            resize_ratio = new_width / frame_width
            new_height = int(resize_ratio * frame_height)
        else:
            new_height = short_side_size
            resize_ratio = new_height / frame_height
            new_width = int(resize_ratio * frame_width)
        for frame in video:
            if show_video and (cv2.waitKey(35) & 0xff != ord('q')):
                cv2.imshow("frame", frame)
            # 1.视频帧转rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2.按照短边尺寸进行resize
            frame = cv2.resize(frame, (new_width, new_height))
            buffers.append(frame)
    # 3.归一化
    buffers = np.array(buffers) / 255
    for _ in range(crop_times):
        # 4.随机crop_times次图像裁剪
        crop_h_start = rd.randint(new_height - crop_size[0]) if new_height - crop_size[0] > 0 else 0
        crop_w_start = rd.randint(new_width - crop_size[1]) if new_width - crop_size[1] > 0 else 0
        buffers_crop = buffers[:, crop_h_start:crop_h_start + crop_size[0], crop_w_start:crop_w_start + crop_size[1], :]
        for i in range(clip_count):
            # 5.针对每次随机图像裁剪，进行clip_count次随机时间维度切取
            clip_result = tempral_clip(buffers_crop)
            clips.append(np.transpose(clip_result, axes=[3, 0, 1, 2]))
    clips = np.array(clips)
    clips = t.from_numpy(clips).type(t.FloatTensor)
    return clips


def predict(video):
    """
    param video: 可以填入视频路径，字符串格式；也可以填入帧流，ndarray格式，形状为[D, H, W, C]
    """
    clips = load_data(video)
    softmax_results = []
    predict_loader = make_predict_loader(clips, batch_size, num_workers)
    for d in predict_loader:
        d_cuda = d.cuda(0)
        with t.no_grad():
            output = model(d_cuda)
            softmax_result = softmax_op(output)
            softmax_results.append(softmax_result)
    softmax_results = t.cat(softmax_results, dim=0).mean(dim=0)
    predict_class_index = t.argmax(softmax_results)
    predict_class_name = class_names[predict_class_index]
    confidences = dict(zip(class_names, softmax_results.detach().cpu().numpy().tolist()))
    return predict_class_name, predict_class_index, confidences


if __name__ == "__main__":
    predict_class_name, predict_class_index, confidences = predict(video_pth)
    print("confidence:", confidences)
    print("action name:", predict_class_name)