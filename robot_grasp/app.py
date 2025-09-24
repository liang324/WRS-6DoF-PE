from flask import Flask, request, jsonify, render_template
from robot_grasp.local_detection_utils import RobotController, execute_robot_with_command, \
    setup_kinectv3_camera_params, kinectv3_capture_and_detect
import sys

app = Flask(__name__)

# 全局变量，但不在这里打开串口
robot_ctrl = None
init_done = False  # 记录是否初始化成功


@app.route("/")
def index():
    """
    前端控制面板
    """
    return render_template("index.html")

# 全局状态变量
robot_ctrl = None
init_done = False     # 已初始化状态
place_done = False    # 已放置状态
zero_done = False     # 已归零状态


@app.route("/robot/init", methods=["GET", "POST"])
def init_robot():
    """ 初始化机械臂串口连接 """
    global robot_ctrl, init_done, place_done, zero_done

    if robot_ctrl is None:
        try:
            robot_ctrl = RobotController()   # 创建并连接 COM3
            init_done = True
            place_done = False
            zero_done = False
            return jsonify({"status": "ok", "msg": "机械臂初始化成功"})
        except Exception as e:
            return jsonify({"status": "error", "msg": f"初始化失败: {e}"})
    else:
        return jsonify({"status": "ok", "msg": "机械臂已初始化"})


@app.route("/robot/cmd", methods=["GET", "POST"])
def robot_command():
    """ 接收前端机械臂指令并执行 """
    global robot_ctrl, init_done, place_done, zero_done

    if robot_ctrl is None:
        return jsonify({"status": "error", "msg": "机械臂未初始化，请先调用 /robot/init"})

    # 获取 cmd 参数
    if request.method == "GET":
        cmd = request.args.get("cmd")
    else:
        data = request.get_json()
        cmd = data.get("cmd") if data else None

    if not cmd:
        return jsonify({"status": "error", "msg": "缺少 cmd 参数"})

    # 00: 初始化（有些场景需要重新初始化）
    if cmd == "00":
        message = "[INFO] 正在执行机械臂初始化..."
        try:
            if robot_ctrl.go_init():  # 初始化动作
                init_done = True
                place_done = False
                zero_done = False
                message += " 初始化完成"
                return jsonify({
                    "status": "ok",
                    "init_done": init_done,
                    "place_done": place_done,
                    "zero_done": zero_done,
                    "msg": message
                })
            else:
                message += " 初始化失败"
                return jsonify({"status": "error", "msg": message})
        except Exception as e:
            return jsonify({"status": "error", "msg": f"初始化异常: {e}"})

    # 归零（必须先初始化）
    elif cmd == "99":
        if not init_done:
            return jsonify({"status": "error", "msg": "归零前必须先初始化"})
        message = "[INFO] 正在执行机械臂归零操作..."
        try:
            if robot_ctrl.go_zero():  # 你需要在 RobotController 类中定义 go_zero() 方法
                zero_done = True
                message += " 归零操作完成"
                return jsonify({
                    "status": "ok",
                    "init_done": init_done,
                    "place_done": place_done,
                    "zero_done": zero_done,
                    "msg": message
                })
            else:
                message += " 归零操作失败"
                return jsonify({"status": "error", "msg": message})
        except Exception as e:
            return jsonify({"status": "error", "msg": f"归零操作异常: {e}"})

    # 11: 放置任务（必须先初始化）
    elif cmd == "11":
        if not init_done:
            return jsonify({"status": "error", "msg": "机械臂未初始化"})
        message = "[INFO] 正在执行放置任务..."
        try:
            if robot_ctrl.go_place():
                place_done = True
                message += " 放置完成"
                return jsonify({
                    "status": "ok",
                    "init_done": init_done,
                    "place_done": place_done,
                    "zero_done": zero_done,
                    "msg": message
                })
            else:
                message += " 放置失败"
                return jsonify({"status": "error", "msg": message})
        except Exception as e:
            return jsonify({"status": "error", "msg": f"放置异常: {e}"})
    else:
        return jsonify({"status": "error", "msg": f"未知指令: {cmd}"})

    return jsonify({
        "status": "ok",
        "init_done": init_done,
        "place_done": place_done,
        "msg": message,
        # "image_path": image_path if cmd == "22" else None
    })


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # 可以防止奇怪的重复启动和死锁
    app.run(host="0.0.0.0", port=6050, debug=True)
