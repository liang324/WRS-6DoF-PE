# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/manual_scan.py

import argparse
import datetime
import os
import open3d as o3d


class RecorderWithCallback:
    """
    使用回调函数的Azure Kinect录制器类.  该类负责从Azure Kinect传感器录制数据,并将其保存为彩色图像、深度图像和可选的点云文件.
    """

    def __init__(self, config, device, base_output_dir, align_depth_to_color, arg):
        """
        初始化RecorderWithCallback对象.

        Args:
            config: Azure Kinect传感器配置.
            device:  Azure Kinect设备ID.
            base_output_dir:  输出文件的基本名称（目录的绝对路径）.
            align_depth_to_color: 是否将深度图像与彩色图像对齐.
            arg: 命令行参数对象.
        """
        # 全局标志
        self.flag_exit = False  # 退出标志
        self.flag_record = True  # 录制标志，设置为True自动开始
        self.output = base_output_dir
        self.save_ply = arg.save_ply  # 是否保存ply点云文件
        self.align_depth_to_color = align_depth_to_color  # 是否对齐深度和彩色图像
        print("设备ID:", device)  # 打印设备ID
        self.recorder = o3d.io.AzureKinectRecorder(config, device)  # 创建Azure Kinect录制器对象

        if not self.recorder.init_sensor():  # 初始化传感器
            raise RuntimeError('无法连接到传感器')

    def save_file(self, save_filename, save_rgbd, save_idx):
        """
        保存彩色图像、深度图像和可选的点云文件

        Args:
            save_filename:  时间戳文件对象
            save_rgbd:  包含彩色和深度图像数据的RGBD图像对象
            save_idx:  帧索引
        """
        # 保存时间戳（修复timer问题）
        current_time = datetime.datetime.now().timestamp()
        f_time = f"{save_idx}  {current_time}\n"
        save_filename.write(f_time)

        # 修复文件路径构造问题
        # 保存彩色图像
        color_filename = os.path.join(self.output, 'color', f'{save_idx:05d}.jpg')
        print('正在写入', color_filename)
        o3d.io.write_image(color_filename, save_rgbd.color)

        # 保存深度图像
        depth_filename = os.path.join(self.output, 'depth', f'{save_idx:05d}.png')
        print('正在写入', depth_filename)
        o3d.io.write_image(depth_filename, save_rgbd.depth)

        ply_filename = None
        if self.save_ply:  # 保存点云
            try:
                # 尝试读取内参文件
                intrinsic_file = 'intrinsic_3072p.json'
                if os.path.exists(intrinsic_file):
                    intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_file)
                else:
                    # 如果没有内参文件，使用默认内参
                    print(f"警告: 找不到内参文件 {intrinsic_file}，使用默认内参")
                    intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

                img_depth = o3d.geometry.Image(save_rgbd.depth)
                img_color = o3d.geometry.Image(save_rgbd.color)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    img_color, img_depth, convert_rgb_to_intensity=False)
                ply = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

                ply_filename = os.path.join(self.output, 'ply', f'{save_idx:05d}.ply')
                print('正在写入', ply_filename)
                o3d.io.write_point_cloud(ply_filename, ply)
            except Exception as e:
                print(f"保存点云时出错: {e}")
                ply_filename = None

        return color_filename, depth_filename, ply_filename

    def run(self, chosen_mode, fps_rate, output_sub_dir_name=None):
        """
        运行自动拍摄
        """
        # 目录准备逻辑
        if output_sub_dir_name is not None:
            final_output_dir = os.path.join(self.output, output_sub_dir_name)
        else:
            timestamp_dir = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())
            final_output_dir = os.path.join(self.output, timestamp_dir)

        self.output = final_output_dir
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'color'), exist_ok=True)
        if self.save_ply:
            os.makedirs(os.path.join(self.output, 'ply'), exist_ok=True)

        f = open(os.path.join(self.output, "timestamp.txt"), "w")
        returned_color_path, returned_depth_path, returned_ply_path = None, None, None

        print("正在自动拍摄...")

        # 等待几帧让传感器稳定
        for i in range(5):
            rgbd = self.recorder.record_frame(True, self.align_depth_to_color)
            if rgbd is None:
                print(f"等待传感器稳定... {i + 1}/5")
                continue

        # 自动拍摄一帧
        rgbd = self.recorder.record_frame(self.flag_record, self.align_depth_to_color)
        print("rgbd数据:", rgbd)

        if rgbd is not None:
            returned_color_path, returned_depth_path, returned_ply_path = self.save_file(f, rgbd, 0)
            print("自动拍摄并保存完成:", self.output)
        else:
            print("拍摄失败，未获取到有效数据。")

        f.close()
        self.recorder.close_record()
        return returned_color_path, returned_depth_path, returned_ply_path


if __name__ == '__main__':
    # 获取当前文件所在目录的绝对路径
    current_dir_path = os.path.abspath(os.path.dirname(__file__))
    print("当前文件所在目录的绝对路径:", current_dir_path)

    parser = argparse.ArgumentParser(description='Azure Kinect 自动拍摄器.')
    parser.add_argument('--config', default='default_config.json', type=str, help='输入 Azure Kinect 配置文件')
    parser.add_argument('--save_ply', action='store_true', help='输出 ply 点云文件')  # 修复为action='store_true'
    parser.add_argument('--output', default='kinect_data', type=str, help='输出文件名（目录）,相对于脚本所在目录')
    parser.add_argument('--list', action='store_true', help='列出可用的 Azure Kinect 传感器')
    parser.add_argument('--device', type=int, default=0, help='输入 Kinect 设备 ID')
    parser.add_argument('--align_depth_to_color', action='store_true',
                        help='启用深度图像与彩色图像对齐')  # 修复为action='store_true'
    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"警告: 配置文件 {args.config} 不存在，使用默认配置")
        config = o3d.io.AzureKinectSensorConfig()
    else:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)

    base_data_root_dir = current_dir_path
    print(f'所有数据将存储在基准目录: {base_data_root_dir} 下的子目录中.')

    device = args.device
    if device < 0 or device > 255:
        print('不支持的设备 ID,回退到 0')
        device = 0

    try:
        record_fps = 30
        r = RecorderWithCallback(config, device, base_data_root_dir, args.align_depth_to_color, args)
        color_path, depth_path, ply_path = r.run("auto", record_fps, args.output)

        print("\n=== 拍摄完成 ===")
        print("捕获的 RGB 图:", color_path)
        print("捕获的深度图:", depth_path)
        if ply_path:
            print("捕获的点云:", ply_path)
        else:
            print("未生成点云文件")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()