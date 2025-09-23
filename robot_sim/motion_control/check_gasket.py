import os
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from PIL import Image
from typing import Literal
from ultralytics import YOLO

# from Detecting_location_gofa5_class import run_class

import basis.robot_math as rm


class GasketDetector:
    def __init__(self, camera_type: Literal['d405', 'd435'] = 'd405', aruco_dict=None, aruco_params=None,
                 save_directory='Data_Intel_Realsense_d405'):
        marker_0_pos = np.array([0, 0])
        marker_1_pos = marker_0_pos + np.array([0.23, 0])
        marker_2_pos = marker_0_pos + np.array([0, 0.17])
        marker_3_pos = marker_0_pos + np.array([0.23, 0.17])
        self.marker_real_coords = {0: marker_0_pos, 1: marker_1_pos, 2: marker_2_pos, 3: marker_3_pos}
        self.camera_type = camera_type
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)
        if self.camera_type.lower() == 'd405':
            self.camera_matrix = np.array([[434.44, 0., 322.235],
                                           [0., 433.249, 236.842],
                                           [0., 0., 1.]])
            self.dist_coeffs = np.array([[-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724]])
            self.depth_scale = 9.999999747378752e-05
        elif self.camera_type.lower() == 'd435':
            self.camera_matrix = np.array([[606.627, 0., 324.281],
                                           [0., 606.657, 241.149],
                                           [0., 0., 1.]])
            self.dist_coeffs = np.array([0., 0., 0., 0., 0.])
            self.depth_scale = 0.0010000000474974513
        else:
            raise ValueError('Either d405 or d435')
        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco_params or aruco.DetectorParameters()

    def capture_from_camera(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        last_saved_color_path = None

        try:
            print("相机启动,按 Enter 键拍照,按 q 退出...")
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                cv2.imshow('Color Image', color_image)
                key = cv2.waitKey(1)

                if key == 13:  # Enter 键
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    color_path = os.path.join(self.save_directory, f'color_image_{timestamp}.jpg')
                    # depth_path = os.path.join(self.save_directory, f'depth_image_{timestamp}.png')

                    cv2.imwrite(color_path, color_image)
                    # Image.fromarray(depth_image.astype(np.uint16)).save(depth_path)

                    print(f"已保存图像：{color_path}")
                    # print(f"已保存深度图：{depth_path}")

                    last_saved_color_path = color_path

                if key & 0xFF == ord('q'):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

        if last_saved_color_path:
            print(f"使用最后保存的图像：{last_saved_color_path}")
            image = cv2.imread(last_saved_color_path)
            return image, last_saved_color_path
        else:
            print("没有保存图像")
            return None, None

    @staticmethod
    def capture_single_image(warmup_frames=30):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        align = rs.align(rs.stream.color)

        try:
            pipeline.start(config)

            # 预热：丢弃前 warmup_frames 帧,确保白平衡等稳定
            for _ in range(warmup_frames):
                frames = pipeline.wait_for_frames()

            # 获取对齐后的帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                raise RuntimeError("未获取到有效帧")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image
        finally:
            pipeline.stop()

    def detect_aruco_pixels(self, image, draw=True):
        h, w = image.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), alpha=1)
        map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_K, (w, h),
                                                 cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        if draw:
            cv2.imshow('undistorted image', undistorted_img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            print("未检测到任何 ArUco 标记")
            return None

        ids = ids.flatten()
        marker_px = {}
        for i, marker_id in enumerate(ids):
            if marker_id not in [0, 1, 2, 3]:
                continue  # 排除非目标ID

            center = np.mean(corners[i][0], axis=0)
            marker_px[marker_id] = center

            if draw:
                cv2.circle(image, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                cv2.putText(image, str(marker_id), tuple(center.astype(int) + np.array([5, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(marker_px) < 4:
            print("未检测到全部四个 marker（需要 ID 0–3）")
            return None

        aruco.drawDetectedMarkers(image, corners, ids)
        return marker_px, image, undistorted_img

    @staticmethod
    def warp_to_marker_frame(image, aruco_px_dict, output_size=(460, 340)):
        """
        使用四个 ArUco 标记进行透视变换,将 marker 围成的区域变为矩形.
        默认顺序为：
            ID 0 - 左上
            ID 1 - 右上
            ID 2 - 左下
            ID 3 - 右下
        """
        try:
            p0 = np.array(aruco_px_dict[0], dtype=np.float32)  # 左上
            p1 = np.array(aruco_px_dict[1], dtype=np.float32)  # 右上
            p2 = np.array(aruco_px_dict[2], dtype=np.float32)  # 左下
            p3 = np.array(aruco_px_dict[3], dtype=np.float32)  # 右下
        except KeyError:
            print("marker ID 不完整,无法进行透视变换")
            return None, None

        # 构造目标矩形区域
        src_pts = np.array([p0, p1, p2, p3], dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [0, output_size[1] - 1],
            [output_size[0] - 1, output_size[1] - 1]
        ], dtype=np.float32)

        # 仿射变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, output_size)

        return warped, M

    def detect_gaskets_pixels(self, image, roi, draw, threshold=70, dp=1.2, minDist=10, param1=100, param2=30,
                              minRadius=5,
                              maxRadius=30):
        x1, x2 = roi['x']
        y1, y2 = roi['y']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[int(y1):int(y2), int(x1):int(x2)] = 255
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.GaussianBlur(masked, (5, 5), 2)
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        centers = []
        if draw:
            output_img = image.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0]:
                center = np.array([x, y], dtype=np.float32)
                centers.append(center)
                if draw:
                    cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)  # 绿色圆
                    cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)  # 红心点

        if draw:
            cv2.imshow('Thresh image', thresh)
            cv2.imshow("Detected Gaskets", output_img)
            cv2.imwrite(f"{self.save_directory}/detected_gasket.jpg", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"检测到 {len(centers)} 个垫片圆形")
        return thresh, centers

    @staticmethod
    def detect_gaskets_by_yolo(image):
        pth_path = r"F:\Study\point cloud\wrs-qiu\wrs-qiu\0000_grasp_concave\yolov11\result\OBB_epochs_50\train\weights\best.pt"  # 加载训练得到的最佳模型权重
        test_image = image
        model = YOLO(pth_path)
        results = model(test_image, conf=0.35)

        for result in results:
            xywhr = result.obb.xywhr  # 中心点 (x,y) 宽高 (w, h) 角度 r(弧度制)
            xyxyxyxy = result.obb.xyxyxyxy  # x1 y1 x2 y2 x3 y3 x4 y4
            corners = xyxyxyxy.cpu().numpy().reshape((-1, 4, 2))  # shape: (4, 2)
            gasket_centers = xywhr.cpu().numpy()[:, :2]
            classes = [result.names[cls.item()] for cls in result.obb.cls.int()]
            result_img = result.plot()
            gasket_centers.tolist()

            return result_img, corners, gasket_centers, classes

    def detect_gaskets_interactive(self, image, roi):
        """
        带交互式参数调节的垫片检测函数
        通过Trackbar实时调整参数并查看效果
        """
        # 初始化参数
        params = {
            'threshold': 70,
            'dp': 12,  # 存储为整数(实际值=dp/10)
            'minDist': 5,
            'param1': 100,
            'param2': 30,
            'minRadius': 5,
            'maxRadius': 30
        }

        # 创建显示窗口
        cv2.namedWindow('Parameters')
        cv2.namedWindow('Detection Result')
        centers = []

        # 创建Trackbar回调函数
        def nothing(x):
            nonlocal centers
            # 获取当前所有参数值
            params['threshold'] = cv2.getTrackbarPos('Threshold', 'Parameters')
            params['dp'] = max(1, cv2.getTrackbarPos('DP(x0.1)', 'Parameters'))
            params['minDist'] = cv2.getTrackbarPos('MinDist', 'Parameters')
            params['param1'] = max(1, cv2.getTrackbarPos('Param1', 'Parameters'))
            params['param2'] = max(1, cv2.getTrackbarPos('Param2', 'Parameters'))
            params['minRadius'] = cv2.getTrackbarPos('MinRadius', 'Parameters')
            params['maxRadius'] = cv2.getTrackbarPos('MaxRadius', 'Parameters')

            # 执行检测
            x1, x2 = roi['x']
            y1, y2 = roi['y']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            mask[int(y1):int(y2), int(x1):int(x2)] = 255
            masked = cv2.bitwise_and(gray, gray, mask=mask)
            blurred = cv2.GaussianBlur(gray, (5, 5), 2)
            _, thresh = cv2.threshold(blurred, params['threshold'], 255, cv2.THRESH_BINARY_INV)

            # 实际dp值为存储值/10
            circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT,
                                       dp=params['dp'] / 10,
                                       minDist=params['minDist'],
                                       param1=params['param1'],
                                       param2=params['param2'],
                                       minRadius=params['minRadius'],
                                       maxRadius=params['maxRadius'])

            # 显示结果
            output_img = image.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (x, y, r) in circles[0]:
                    center = np.array([x, y], dtype=np.float32)
                    centers.append(center)
                    cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)

            # 显示阈值图像和结果
            cv2.imshow('Threshold', thresh)
            cv2.imshow('Detection Result', output_img)
            return centers

        # 创建Trackbars
        cv2.createTrackbar('Threshold', 'Parameters', params['threshold'], 255, nothing)
        cv2.createTrackbar('DP(x0.1)', 'Parameters', params['dp'], 30, nothing)  # dp范围1.0-3.0
        cv2.createTrackbar('MinDist', 'Parameters', params['minDist'], 100, nothing)
        cv2.createTrackbar('Param1', 'Parameters', params['param1'], 200, nothing)
        cv2.createTrackbar('Param2', 'Parameters', params['param2'], 100, nothing)
        cv2.createTrackbar('MinRadius', 'Parameters', params['minRadius'], 100, nothing)
        cv2.createTrackbar('MaxRadius', 'Parameters', params['maxRadius'], 200, nothing)

        # 初始调用一次
        nothing(0)

        # 等待用户操作
        print("调整参数后按ESC退出...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                break

        cv2.destroyAllWindows()
        return centers

    def pixel_to_real_from_affine(self, px, output_size=(460, 340)):
        """在仿射图中计算像素点相对 ID0 的实际物理坐标"""
        dx_real = np.linalg.norm(self.marker_real_coords[1] - self.marker_real_coords[0])  # 通常是 0.23m
        dy_real = np.linalg.norm(self.marker_real_coords[2] - self.marker_real_coords[0])  # 通常是 0.17m
        w, h = output_size

        # 将像素转换为相对于 ID0 的物理坐标
        scale_x = dx_real / w
        scale_y = dy_real / h
        real_x = px[0] * scale_x
        real_y = px[1] * scale_y
        return np.array([real_x, real_y])

    @staticmethod
    def plot_transformed_gasket_and_marker_view(original_image, warped_image, aruco_px_dict, M,
                                                gasket_centers_transformed, classes):
        """
        显示左侧原始图像和右侧透视变换图,并标注 ArUco markers 和 gasket 检测结果
        """

        # 在原图中画 ArUco markers
        orig_img_vis = original_image.copy()
        for marker_id, pt in aruco_px_dict.items():
            pt_int = tuple(pt.astype(int))
            cv2.circle(orig_img_vis, pt_int, 5, (0, 0, 255), -1)
            cv2.circle(orig_img_vis, pt_int, 5, (0, 0, 255), -1)
            cv2.putText(orig_img_vis, f"id:{marker_id}", pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 连线：默认连 id0-id1（x方向）,id0-id2（y方向）
        if all(k in aruco_px_dict for k in [0, 1, 2]):
            cv2.line(orig_img_vis, tuple(aruco_px_dict[0].astype(int)), tuple(aruco_px_dict[1].astype(int)),
                     (0, 255, 0), 2)
            cv2.line(orig_img_vis, tuple(aruco_px_dict[0].astype(int)), tuple(aruco_px_dict[2].astype(int)),
                     (0, 255, 0), 2)

        # --- 绘图 ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # 左图：原图 + ArUco
        axs[0].imshow(cv2.cvtColor(orig_img_vis, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image with ArUco Markers")

        # 右图：变换图 + gasket
        axs[1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Warped Image with Detected Gaskets")
        axs[1].set_xlim([0, warped_image.shape[1]])
        axs[1].set_ylim([warped_image.shape[0], 0])  # 保持左上为原点

        for pt in gasket_centers_transformed:
            axs[1].scatter(pt[0], pt[1], color='green')
            axs[1].text(pt[0] + 5, pt[1] + 5, "Gasket", fontsize=8, color='green')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_pair_pnt_to_split_stack(stack_gasket_corners, offset_pix=50):
        corners = stack_gasket_corners
        d01 = np.linalg.norm(corners[0] - corners[1])
        d12 = np.linalg.norm(corners[1] - corners[2])
        if d01 > d12:
            long_vec = corners[1] - corners[0]
        else:
            long_vec = corners[2] - corners[1]
        perp_vec = np.array([-long_vec[1], long_vec[0]])
        perp_unit = perp_vec / np.linalg.norm(perp_vec)
        center = np.mean(corners, axis=0)
        p1 = center + offset_pix * perp_unit
        p2 = center - offset_pix * perp_unit
        return p1, p2

    @staticmethod
    def image_to_robot(image_coord):
        # 71.25 - 8.75
        image_coord_x = image_coord[0]
        image_coord_y = image_coord[1]
        image_coord_z = 0
        image_coord = np.array([image_coord_x, image_coord_y, image_coord_z])
        convert_pos = np.array([0.7125, -0.0875, 0.033]) + np.array([0.17, 0, 0])
        # convert_pos = np.array([0.668, -0.182, 0]) + np.array([0.215, 0.095, 0.018])
        convert_rot = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-np.pi / 2)
        return image_coord.dot(convert_rot) + convert_pos

    def run_on_image(self, image, draw):
        aruco_px, img_marked, undistorted_img = self.detect_aruco_pixels(image, draw=draw)
        if not aruco_px or not all(k in aruco_px for k in [0, 1, 2, 3]):
            print("缺少构建坐标系所需的 ArUco 标记 (0,1,2,3)")
            return []

        warped_image, M = self.warp_to_marker_frame(image, aruco_px, output_size=(460, 340))
        detect_area = {'x': [30, 430], 'y': [0, 340]}

        # _, gasket_pixels = self.detect_gaskets_pixels(warped_image, detect_area, draw=draw, threshold=100, param2=23)
        _, gasket_corners, gasket_centers, gasket_classes = self.detect_gaskets_by_yolo(warped_image)

        # if len(gasket_pixels) == 0:
        #     for threshold in [95, 90, 85, 80]:
        #         _, gasket_pixels = self.detect_gaskets_pixels(warped_image, detect_area, draw=draw, threshold=threshold,
        #                                                       param2=23)
        #         if len(gasket_pixels) != 0:
        #             break
        print(gasket_centers)

        self.plot_transformed_gasket_and_marker_view(image, warped_image, aruco_px, M, gasket_centers, gasket_classes)

        gasket_world_positions = []
        pair_pnt_to_split_stack_gasket = []

        for i, px in enumerate(gasket_centers):
            if gasket_classes[i] == 'gasketsingle':
                pos_image = self.pixel_to_real_from_affine(px, output_size=(460, 340))
                print(f"相对图像坐标系的位置：{pos_image}")
                pos_robot = self.image_to_robot(pos_image)
                gasket_world_positions.append(pos_robot)
            elif gasket_classes[i] == 'gasketstack':
                pair_pnt = self.get_pair_pnt_to_split_stack(gasket_corners[i], 50)
                pnt1 = self.image_to_robot(self.pixel_to_real_from_affine(pair_pnt[0], output_size=(460, 340)))
                pnt2 = self.image_to_robot(self.pixel_to_real_from_affine(pair_pnt[1], output_size=(460, 340)))
                pair_pnt_to_split_stack_gasket.append([pnt1, pnt2])

        return gasket_world_positions, pair_pnt_to_split_stack_gasket


if __name__ == "__main__":
    def main():
        image = cv2.imread(
            r'F:\Study\point cloud\wrs-qiu\wrs-qiu\0000_grasp_concave\Data_Intel_Realsense_d435\color_image_20250612-095003.jpg')
        if image is None:
            raise FileNotFoundError("图像读取失败,请检查路径")

        camera_type = 'd435'

        detector = GasketDetector(camera_type=camera_type, save_directory=f"Data_Intel_Realsense_{camera_type}")
        image, path = detector.capture_from_camera()
        # image_1 = image.copy()
        image_2 = image.copy()
        # gaskets = run_class(image_1)
        # if gaskets:
        #     print("本地图像坐标下的垫片位置:")
        #     for p in gaskets:
        #         p = p + np.array([0.1375, -0.0375, 0.015])
        #         print(f"({p[0]}, {p[1]}, {p[2]})")

        result = detector.run_on_image(image_2)

        if result:
            print("本地图像坐标下的垫片位置:")
            for p in result:
                print(f"({p[0]}, {p[1]}, {p[2]})")


    main()
