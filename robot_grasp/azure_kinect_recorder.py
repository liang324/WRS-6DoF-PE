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
import timer


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
            # filename:  输出文件的基本名称（目录）.
            base_output_dir:  输出文件的基本名称（目录的绝对路径）.
            align_depth_to_color: 是否将深度图像与彩色图像对齐.
            arg: 命令行参数对象.
        """
        # 全局标志
        self.flag_exit = False  # 退出标志
        self.flag_record = False  # 录制标志
        self.output = base_output_dir
        self.save_ply = arg.save_ply  # 是否保存ply点云文件
        self.align_depth_to_color = align_depth_to_color  # 是否对齐深度和彩色图像
        print(device)  # 打印设备ID
        self.recorder = o3d.io.AzureKinectRecorder(config, device)  # 创建Azure Kinect录制器对象
        self.timer = timer.Timer()  # 创建计时器对象
        if not self.recorder.init_sensor():  # 初始化传感器
            raise RuntimeError('无法连接到传感器')

    def escape_callback(self, vis):
        """
        ESC键回调函数,用于停止录制并退出.
        """
        self.flag_exit = True  # 设置退出标志
        if self.recorder.is_record_created():  # 检查是否已创建录制文件
            print('录制完成.')
        else:
            print('未录制任何内容.')
        return False

    def space_callback(self, vis):
        """
        空格键回调函数,用于暂停/继续录制.
        """
        if self.flag_record:  # 如果正在录制
            print('录制已暂停. 按[空格]键继续. 按[ESC]键保存并退出.')
            self.flag_record = False  # 暂停录制
        else:  # 如果已暂停
            print('录制已恢复,视频可能不连续. 按[空格]键暂停. 按[ESC]键保存并退出.')
            self.flag_record = True  # 继续录制
        return False

    def save_file(self, save_filename, save_rgbd, save_idx):
        """
        保存彩色图像、深度图像和可选的点云文件

        Args:
            save_filename:  时间戳文件对象
            save_rgbd:  包含彩色和深度图像数据的RGBD图像对象
            save_idx:  帧索引
        """
        # 保存时间戳
        f_time = str(save_idx) + '  ' + str(datetime.datetime.now().timestamp()) + ' ' + str(
            self.timer.counter()) + '\n'
        save_filename.write(f_time)

        # 保存彩色图像
        color_filename = os.path.join(self.output, '{0}/color/{1:05d}.jpg'.format(self.output, save_idx))
        print('正在写入 {}'.format(color_filename))
        o3d.io.write_image(color_filename, save_rgbd.color)

        # 保存深度图像
        depth_filename = os.path.join(self.output, '{0}/depth/{1:05d}.png'.format(self.output, save_idx))
        print('正在写入 {}'.format(depth_filename))
        o3d.io.write_image(depth_filename, save_rgbd.depth)

        if self.save_ply:  # 保存点云
            intrinsic = o3d.io.read_pinhole_camera_intrinsic('intrinsic_3072p.json')  # 读取相机内参
            img_depth = o3d.geometry.Image(save_rgbd.depth)
            img_color = o3d.geometry.Image(save_rgbd.color)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth,
                                                                            convert_rgb_to_intensity=False)
            ply = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            ply_filename = os.path.join(self.output, '{0}/ply/{1:05d}.ply'.format(self.output, save_idx))
            print('正在写入 {}'.format(ply_filename))
            o3d.io.write_point_cloud(ply_filename, ply)
            o3d.visualization.draw_geometries([ply])

        # return save_rgbd.color, save_rgbd.depth, ply
        return color_filename, depth_filename, ply_filename

    def run(self, chosen_mode, fps_rate, output_sub_dir_name=None):
        """
        运行录制器,从Azure Kinect传感器录制数据.

        Args:
            chosen_mode: 录制模式,'manual' 或 'auto'.
            fps_rate:  自动模式下的帧率.
        """
        glfw_key_escape = 256  # ESC键
        glfw_key_space = 32  # 空格键
        vis = o3d.visualization.VisualizerWithKeyCallback()  # 创建可视化窗口
        vis.register_key_callback(glfw_key_escape, self.escape_callback)  # 注册ESC键回调
        vis.register_key_callback(glfw_key_space, self.space_callback)  # 注册空格键回调
        vis.create_window('录制器', 1920, 540)  # 创建窗口
        print("录制器已初始化. 按[空格]键开始. 按[ESC]键保存并退出.")

        if output_sub_dir_name is not None:
            # 如果提供了子目录名,则将其与基准路径合并
            final_output_dir = os.path.join(self.output, output_sub_dir_name)
        else:
            # 如果没有提供,则使用时间戳作为子目录名
            timestamp_dir = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())
            final_output_dir = os.path.join(self.output, timestamp_dir)

        self.output = final_output_dir
        print(f"所有数据将保存到: {self.output}")

        # 创建输出目录及子目录
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(os.path.join(self.output, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'color'), exist_ok=True)
        if self.save_ply:
            os.makedirs(os.path.join(self.output, 'ply'), exist_ok=True)

        vis_geometry_added = False  # 是否已添加几何体到可视化窗口
        f = open(os.path.join(self.output, "timestamp.txt"), "w")  # 打开时间戳文件

        idx = 0  # 帧索引
        time_last = datetime.datetime.now().timestamp()  # 上一帧时间戳
        returned_color_path, returned_depth_path, returned_ply_path = None, None, None  # 初始化返回值

        while not self.flag_exit:  # 循环录制,直到按下ESC键
            rgbd = self.recorder.record_frame(self.flag_record, self.align_depth_to_color)  # 录制一帧数据

            if rgbd is None:  # 如果录制失败
                continue

            if not vis_geometry_added:  # 如果几何体未添加到可视化窗口
                vis.add_geometry(rgbd)  # 添加几何体
                vis_geometry_added = True

            vis.update_geometry(rgbd)  # 更新几何体
            vis.poll_events()  # 处理事件
            vis.update_renderer()  # 更新渲染器

            if self.flag_record:  # 如果正在录制
                if chosen_mode == 'manual':  # 手动模式
                    returned_color_path, returned_depth_path, returned_ply_path = self.save_file(f, rgbd, idx)  # 保存数据
                    idx = idx + 1  # 增加帧索引
                    self.flag_record = False  # 暂停录制
                    break

                if chosen_mode == "auto":  # 自动模式
                    time_now = datetime.datetime.now().timestamp()  # 获取当前时间戳
                    if time_now - time_last >= 1 / fps_rate:  # 检查是否达到帧率
                        print('以', fps_rate, 'fps录制图像,按空格键暂停,按esc键停止录制')
                        returned_color, returned_depth, returned_ply = self.save_file(f, rgbd, idx)  # 保存数据
                        idx = idx + 1  # 增加帧索引
                        time_last = time_now  # 更新上一帧时间戳
                        break

        f.close()  # 关闭时间戳文件
        self.recorder.close_record()  # 关闭录制器

        return returned_color_path, returned_depth_path, returned_ply_path  # 返回 RGB 图、深度图和点云


if __name__ == '__main__':
    # 获取当前文件所在目录的绝对路径
    current_dir_path = os.path.abspath(os.path.dirname(__file__))
    print("当前文件所在目录的绝对路径:", current_dir_path)

    # print(os.environ.get('PATH'))
    parser = argparse.ArgumentParser(description='Azure Kinect MKV 录制器.')
    parser.add_argument('--config', default='default_config.json', type=str, help='输入 Azure Kinect 配置文件')
    # parser.add_argument('--save_ply', action='store_true', default=True, help='输出 ply 点云文件')
    parser.add_argument('--save_ply', default=False , help='输出 ply 点云文件')
    parser.add_argument('--output', default='kinect_data', type=str, help='输出文件名（目录）,相对于脚本所在目录')
    # parser.add_argument('--output', default='kinect_data', type=str, help='输出文件名（目录）')
    parser.add_argument('--list', action='store_true', help='列出可用的 Azure Kinect 传感器')
    parser.add_argument('--device', type=int, default=0, help='输入 Kinect 设备 ID')
    # parser.add_argument('--align_depth_to_color', action='store_true', help='启用深度图像与彩色图像对齐')
    parser.add_argument('--align_depth_to_color', default=True, help='启用深度图像与彩色图像对齐')
    args = parser.parse_args()
    # print(f"Output directory: {args.output}")

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    config = o3d.io.read_azure_kinect_sensor_config(args.config)

    base_data_root_dir = current_dir_path

    print(f'所有数据将存储在基准目录: {base_data_root_dir} 下的子目录中.')

    if args.output is not None:
        filename = args.output
    else:
        filename = '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())
    print('准备写入 {}'.format(filename))

    # 创建输出目录及子目录
    os.makedirs(filename, exist_ok=True)
    os.makedirs(os.path.join(filename, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(filename, 'color'), exist_ok=True)
    if args.save_ply:
        os.makedirs(os.path.join(filename, 'ply'), exist_ok=True)

    device = args.device
    if device < 0 or device > 255:
        print('不支持的设备 ID,回退到 0')
        device = 0
    mode_chosen = False
    while not mode_chosen:
        mode = input("输入录制模式,manual 或 auto：\n")
        if mode == "manual" or mode == "auto":
            mode_chosen = True
        else:
            print("选择的模式错误\n")

    if mode == "auto":
        fps_input = False
        while not fps_input:
            try:
                record_fps = float(input("输入录制帧率 (0 < fps <= 5)：\n"))
                if 0 < record_fps <= 5:
                    fps_input = True
                else:
                    print("帧率输入错误")
            except ValueError:
                print("请输入有效的数字")
    else:
        record_fps = -1

    r = RecorderWithCallback(config, device, base_data_root_dir, args.align_depth_to_color, args)
    color_path, depth_path, ply_path = r.run(mode, record_fps, args.output)

    print("捕获的 RGB 图:", color_path)
    print("捕获的深度图:", depth_path)
    print("捕获的点云:", ply_path)
