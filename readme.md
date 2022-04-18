一.数据集准备
数据格式与UCF-101相同如下:
train_data_root_dir:
    class1
        1.avi
        2.avi
        ...
    class2
        1.avi
        2.avi
        ...
    ...
验证集目录与训练集目录格式相同，class1，class2为类别名称，需与conf.json中的class_names保持一致

二.模型训练
1.配置conf.json中的train以及common部分，common部分表示predict与train公用的（参数值保持一致），参数含义如下：
clip_len：一段视频中沿着时间轴截取一段视频片段，截取视频片段的帧数，也就是送入网络训练数据的时间轴长度
slow_tao：SlowFast Networks中slowpath的时间维度采样步长
alph：slowpath时间维度采样步长与fastpath时间维度采样步长的比值
is_group_conv：是否使用分组卷积
class_names：视频类别名称列表
num_workers：加载数据线程数目
epoch：训练轮数
batch_size：数据加载batch size
CUDA_VISIBLE_DEVICES：调用那些gpu
init_lr：初始学习率
lr_de_epoch：多少个epoch进行一次学习率调整
lr_de_rate：学习率调整率，是上一次学习率的多少倍
train_data_root_dir：训练集根目录路径
valid_data_root_dir：验证集根目录路径
weight_decay：optimizer参数，防止过拟合
short_side_size_range：视频帧resize时，帧短边的resize尺寸范围，从中随机取值对帧短边进行resize
crop_size：视频帧进行空间维度裁剪的尺寸
print_step：输出训练信息的步长间隔
2.运行train.py
```
python train.py
```

三.模型预测
1.配置conf.json中predict的部分，参数含义如下：
video_pth：当use_camera为false时对本地视频进行预测，这里指定的是本地视频路径
clip_count：对要识别的视频进行多少次时间维度的截取（用于预测结果投票）
short_side_size：视频帧短边resize尺寸
crop_size：视频帧的crop尺寸，[crop_h, crop_w]
crop_times：进行多少次crop（用于预测结果投票）
batch_size：与测时的batch_size多大（因为最终要用投票的方式进行预测，因此一段视频要截取crop_times * clip_count段clip，送入模型进行预测后取平均）
use_best_model：True使用历史最好模型进行预测，False使用训练最后一个epoch得到的模型进行预测
show_video：是否展示读取的视频，当use_camera为true时候自动指定为false
predict_camera_frame_count：use_camera为true时调用摄像头实时识别时，帧采样时间滑动窗口长度
use_camera：true调用摄像头实时预测，false则读取本地视频进行预测
2.运行predict.py
```
python predict.py
```

