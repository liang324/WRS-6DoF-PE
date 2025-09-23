import argparse
import os
import platform
import random
import sys
import time
from pathlib import Path
import csv
import torch
from torch.backends import cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, MyLoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync


class YoloOpt:
    def __init__(self, weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt',
                 imgsz=(640, 640), conf_thres=0.65,
                 iou_thres=0.65, device='', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False,
                 project='/detect/result', name='result_exp',
                 save_csv=True, csv_path='result.csv'):
        self.weights = weights  # 权重文件地址
        self.source = None  # 待识别的图像
        if imgsz is None:
            self.imgsz = (640, 640)
        self.imgsz = imgsz  # 输入图片的大小,默认 (640,640)
        self.conf_thres = conf_thres  # object置信度阈值 默认0.25 用在nms中
        self.iou_thres = iou_thres  # 做nms的iou阈值 默认0.45 用在nms中
        self.device = device  # 选择设备,可以为GPU、CPU等
        self.view_img = view_img  # 是否展示预测之后的图片或视频,默认 False
        self.classes = classes  # 只保留一部分的类别,默认是全部保留
        self.agnostic_nms = agnostic_nms  # 是否进行类无关的 NMS,默认 False
        self.augment = augment  # 增强推理,默认 False
        self.update = update  # 是否更新所有模型,默认 False
        self.exist_ok = exist_ok  # 是否允许存在同名项目目录,默认 False
        self.project = project  # 保存项目的路径
        self.name = name  # 保存项目的名称
        self.save_csv = save_csv  # 是否保存到CSV文件
        self.csv_path = csv_path  # CSV文件保存路径


class DetectAPI:
    def __init__(self, weights, imgsz=640, device='', csv_path='result.csv'):
        # 初始化参数,加载模型
        self.opt = YoloOpt(weights=weights, imgsz=imgsz, device=device, csv_path=csv_path)
        weights = self.opt.weights  # 传入的权重
        imgsz = self.opt.imgsz  # 传入的图像尺寸

        # Initialize 初始化
        self.device = select_device(self.opt.device)  # 选择设备,GPU 或 CPU
        self.half = False  # 强制禁用FP16推理,使用FP32

        # 加载模型
        self.model = DetectMultiBackend(weights, self.device, dnn=False)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # 读取类别名称和颜色
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # 初始化CSV保存路径
        self.csv_path = csv_path
        if self.opt.save_csv:
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame', 'Class', 'Confidence', 'x1', 'y1', 'x2', 'y2'])

    def detect(self, source, frame_rate=30):
        # 输入 detect([img])
        if type(source) != list:
            raise TypeError('source must be a list containing images read by cv2')

        dataset = MyLoadImages(source)
        bs = 1  # batch size

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # 模型预热

        dt, seen = (Profile(), Profile(), Profile()), 0
        frame_count = 0  # 用于标记当前帧数

        # # 用于统计每种类别的数量
        # class_counts = {i: 0 for i in range(len(self.names))}  # 使用索引初始化类别统计字典

        # 只统计总的物体数量,不按类别统计
        total_objects = 0

        for im, im0s in dataset:
            frame_count += 1
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)

                # 强制使用FP32单精度浮点数
                im = im.float()

                im /= 255.0  # 归一化到 [0, 1]

                if len(im.shape) == 3:
                    im = im[None]  # 扩展维度

                # 推理
                pred = self.model(im, augment=self.opt.augment)[0]

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                               self.opt.agnostic_nms, max_det=1000)

                # 处理检测结果
                det = pred[0]  # 因为一次处理一张图片,直接取第一个
                im0 = im0s.copy()

                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])

                        # # 更新统计
                        # class_index = int(cls)  # 获取类别的索引
                        # class_counts[class_index] += 1  # 通过索引更新计数

                        # 更新总物体数量
                        total_objects += 1  # 每检测到一个物体,增加总物体数量

                        # 将检测结果保存到CSV文件
                        if self.opt.save_csv:
                            with open(self.csv_path, mode='a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [frame_count, self.names[int(cls)], f'{conf:.2f}', *[int(coord) for coord in xyxy]])

        return im0, pred, total_objects


# 定义一个简单的测试函数,并传入设备参数
def test_yolo(device='0'):
    detect_api = DetectAPI(
        weights=r'D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_DH76\best_rack510.pt',
        imgsz=640, device=device,
        csv_path='detection_results.csv')
    # 使用实际存在的图片路径
    img_list = [cv2.imread(
        r'D:\AI\ABB_wrs_hu_sam\suyixuan_sam\Human_Robot_Collaboration\Tube_Grasping\work3_HRC_Projection_DH76\realsense_data_rack510\pcd\color_20250910_174107.png'),
                # cv2.imread(r'E:\ABB\segment-anything\data\images\00001.jpg')
                ]
    # detect_api.detect(img_list)

    # 获取每张图片的检测物体数量
    for i, img in enumerate(img_list):
        im0, pred, total_objects = detect_api.detect([img])  # 传递图像进行检测
        print(f'Image {i + 1} class counts: {total_objects}')


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    test_yolo(device=args.device)
    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f} seconds')
