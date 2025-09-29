import os
import shutil
import time
import pyrealsense2 as rs
import cv2

from flask import Flask, request, jsonify, render_template, url_for
import sys
from robot_grasp.local_detection_utils import (
    RobotController,
    execute_robot_with_command,
    setup_kinectv3_camera_params,
    kinectv3_capture_and_detect, capture_kinect_data, detect_and_localize
)

app = Flask(__name__)


def api_response(status, msg, kinect_image_url=None, yolo_image_url=None, **kwargs):
    """统一 JSON 返回结构，支持两张图片"""
    resp = {
        "status": status,
        "msg": msg,
        "kinect_image_url": kinect_image_url,
        "yolo_image_url": yolo_image_url
    }
    resp.update(kwargs)
    return jsonify(resp)


# 全局状态变量
robot_ctrl = None
init_done = False  # 机械臂初始化状态（串口连接状态）
go_init_done = False  # 机械臂归零状态
move_done = False  # 机械臂移动状态
kinect_capture_done = False  # Kinect 拍摄状态
yolo_detect_done = False  # kinect_yolo 目标检测状态
yolo_results = None


@app.route("/")
def index():
    """ 前端控制面板 """
    return render_template("index.html")


@app.route("/robot/init", methods=["GET", "POST"])
def init_robot():
    """ 初始化机械臂串口连接 """
    global robot_ctrl, init_done, go_init_done, move_done

    if robot_ctrl is None:
        try:
            robot_ctrl = RobotController()  # 创建并连接 COM3
            init_done = True
            go_init_done = False
            move_done = False
            return api_response("ok", "机械臂初始化完成")

        except Exception as e:
            return api_response("error", f"机械臂初始化失败: {e}")
    else:
        return api_response("ok", "机械臂已初始化")


# -------------------------
# 机械臂复位
# -------------------------
@app.route("/robot/reset", methods=["POST"])
def robot_reset():
    global init_done, go_init_done, move_done

    if not init_done:
        return api_response("error", "请先执行机械臂初始化")
    message = "[INFO] 正在执行机械臂复位任务..."
    try:
        if robot_ctrl.go_init():  # 机械臂复位
            go_init_done = True
            move_done = False
            message += " 机械臂复位任务完成"
            return api_response("ok", message)
        else:
            message += " 机械臂复位任务失败"
            return api_response("error", message)
    except Exception as e:
        return api_response("error", f"机械臂复位任务异常: {e}")


# -------------------------
# 机械臂移动
# -------------------------
@app.route("/robot/move", methods=["POST"])
def robot_move():
    global init_done, go_init_done, move_done

    if not go_init_done:
        return api_response("error", "必须先完成机械臂复位任务才能执行移动任务")
    message = "[INFO] 正在执行机械臂移动任务..."
    try:
        if robot_ctrl.go_place():
            move_done = True
            message += " 机械臂移动任务完成"
            return api_response("ok", message)
        else:
            message += " 机械臂移动任务失败"
            return api_response("error", message)
    except Exception as e:
        return api_response("error", f"机械臂移动任务异常: {e}")


# -------------------------
# Kinect 拍摄
# -------------------------
@app.route("/kinect/capture", methods=["POST"])
def kinect_capture():
    global move_done, kinect_capture_done

    if not move_done:
        return api_response("error", "必须先完成机械臂移动任务，才能执行 Kinect 拍摄任务")

    message = "[INFO] 正在执行kinect拍摄任务..."
    depth_intrin, T_camera_to_robot, depth_scale, save_folder = setup_kinectv3_camera_params()
    success, color_path, depth_path = capture_kinect_data(save_folder)

    if not success:
        return api_response("error", "Kinect 拍摄失败")
    else:
        kinect_capture_done = True
        message += " kinect拍摄任务成功"

    # 将拍摄的彩色图拷贝 / 保存到 Flask 静态目录
    static_img_dir = os.path.join("static", "kinect_images")
    os.makedirs(static_img_dir, exist_ok=True)
    img_filename = os.path.basename(color_path)
    target_path = os.path.join(static_img_dir, img_filename)
    shutil.copy(color_path, target_path)

    # 生成前端可访问的 URL
    kinect_image_url = url_for("static", filename=f"kinect_images/{img_filename}", _external=False)

    return api_response("ok",
                        message,
                        kinect_image_url=kinect_image_url,
                        color_path=color_path,
                        depth_path=depth_path,
                        depth_intrin={
                            "width": depth_intrin.width,
                            "height": depth_intrin.height,
                            "ppx": depth_intrin.ppx,
                            "ppy": depth_intrin.ppy,
                            "fx": depth_intrin.fx,
                            "fy": depth_intrin.fy,
                            "model": int(depth_intrin.model),
                            "coeffs": list(depth_intrin.coeffs)
                        },
                        T_camera_to_robot=T_camera_to_robot.tolist(),
                        depth_scale=float(depth_scale)
                        )


# -------------------------
# Kinect YOLO 检测
# -------------------------
@app.route("/kinect/yolo_detect", methods=["POST"])
def yolo_detect():
    global kinect_capture_done, yolo_detect_done, yolo_results

    if not kinect_capture_done:
        return api_response("error", "必须先完成 Kinect 拍摄任务，才能执行目标检测")

    try:
        message = "[INFO] 正在执行 Kinect YOLO 目标检测任务..."
        data = request.get_json(force=True)
        kinect_data = data.get("kinect")

        color_path = kinect_data.get("color_path")
        depth_path = kinect_data.get("depth_path")
        depth_intrin_dict = kinect_data.get("depth_intrin")

        # 将字典转成 pyrealsense2.intrinsics 对象
        depth_intrin = rs.intrinsics()
        depth_intrin.width = int(depth_intrin_dict["width"])
        depth_intrin.height = int(depth_intrin_dict["height"])
        depth_intrin.ppx = float(depth_intrin_dict["ppx"])
        depth_intrin.ppy = float(depth_intrin_dict["ppy"])
        depth_intrin.fx = float(depth_intrin_dict["fx"])
        depth_intrin.fy = float(depth_intrin_dict["fy"])
        depth_intrin.model = rs.distortion(depth_intrin_dict["model"])
        depth_intrin.coeffs = list(depth_intrin_dict["coeffs"])

        T_camera_to_robot = kinect_data.get("T_camera_to_robot")
        depth_scale = kinect_data.get("depth_scale")

        yolo_results, img_color = detect_and_localize(color_path,
                                                      depth_path,
                                                      depth_intrin,
                                                      T_camera_to_robot,
                                                      depth_scale
                                                      )
        if yolo_results and len(yolo_results) > 0:
            message += " kinect_yolo 目标检测任务成功"
            yolo_detect_done = True

            filename = f"detected_result_{int(time.time())}.jpg"
            output_img_path = os.path.join("static", filename)
            cv2.imwrite(output_img_path, img_color)

            return api_response("ok",
                                message,
                                yolo_image_url=url_for('static', filename=filename),
                                count=len(yolo_results),
                                detections=yolo_results)
        else:
            return api_response("error", "未检测到物体")
    except Exception as e:
        return api_response("error", f"API 调用失败: {e}")


# -------------------------
# 机器人自动搬运接口
# -------------------------
@app.route("/robot/auto_move", methods=["POST"])
def robot_auto_move():
    """
    根据 YOLO 检测结果自动搬运物体，并避开障碍物
    """
    global yolo_detect_done, yolo_results

    if not yolo_detect_done:
        return api_response("error", "必须先完成 Kinect 目标检测任务，才能执行自动搬运任务")

    try:
        message = "[INFO] 正在执行机器人自动搬运任务..."

        # # 获取前端传来的数据
        # data = request.get_json(force=True)
        # results = data.get("detections")  # YOLO 检测结果
        if not yolo_results or len(yolo_results) == 0:
            return api_response("error", "未检测到可搬运的物体")

        # 读取配置
        object_model_map, object_color_map = robot_ctrl.get_object_configs()

        # 构建障碍物
        obstacles = robot_ctrl.build_obstacles(yolo_results, object_model_map, object_color_map)

        # 自动搬运执行
        robot_ctrl.auto_move_all_objects(yolo_results, move_real_flag=True, obstacles=obstacles)

        message += " 搬运任务成功完成"
        return api_response("ok", message, count=len(yolo_results), detections=yolo_results)

    except Exception as e:
        return api_response("error", f"API 调用失败: {e}")


# -------------------------
# 程序入口
# -------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    app.run(host="0.0.0.0", port=6050, debug=True)
