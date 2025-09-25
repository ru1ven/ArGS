

import cv2
import os
import re

# 定义图像文件夹路径和输出视频路径
#image_folder = '/home/cyc/pycharm/lxy/3DGS/PoseGS/outputs/wild_lxy_colored_9/'
image_folder = '/mnt/sda1/lxy/3DGS/wild_lxy/wood_1/rgb/'
video_path = image_folder+'wood_1.mp4'  # 输出视频路径


# 获取文件夹中所有图像文件
images = [img for img in os.listdir(image_folder) if (img.endswith(".png"))]
# images = [img for img in os.listdir(image_folder) if (img.endswith(".png") and 'wood_1' in img)]

# 按文件名中的数字排序
images = sorted(images,key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))


# 确保文件夹不为空
if not images:
    raise ValueError("图像文件夹为空或没有有效的图像文件！")

# 读取第一张图像以获取宽度和高度
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
height, width, layers = frame.shape

# 定义视频编码器和帧率
fps = 5  # 每秒30帧
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 逐帧将图像写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    video.write(frame)  # 写入当前帧

# 释放资源
video.release()
print(f"视频已保存到: {video_path}")