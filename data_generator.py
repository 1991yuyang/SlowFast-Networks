import cv2
import os
# 制作自己的数据集


data_root_dir = r"/home/yuyang/data/self_act_rec_data/valid_data"  # 数据根目录
class_name = "calling"  # 动作类别名称
class_dir = os.path.join(data_root_dir, class_name)
if not os.path.exists(class_dir):
    os.mkdir(class_dir)
count = len(os.listdir(class_dir))


video_capture = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter(os.path.join(class_dir, '%d.avi' % (count,)), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), video_capture.get(cv2.CAP_PROP_FPS), (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
# cv2.VideoWriter 分别传入的参数是路径 格式，帧率，视频尺寸
success,frame = video_capture.read()
# 成功打开摄像头 直到按esc退出保存视频
while success and not cv2.waitKey(1) == 27:
    video_writer.write(frame)
    cv2.imshow("Video", frame)
    success, frame = video_capture.read()
cv2.destroyWindow('Video')
video_capture.release()

