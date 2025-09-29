// 存储多台相机的拍摄数据
let captureData = {
    kinect: null,
    realsense: null,
    camA: null,
    yolo_results: null,
};


// 更新 UI：状态框 + 信息框 + 图片
function updateUI(data) {
    // 状态框显示完整 JSON（方便调试）
    document.getElementById("status").textContent = JSON.stringify(data, null, 2);

    // 信息框显示 msg
    document.getElementById("info_display").textContent = data.msg || "无消息";

    // 显示 Kinect 原图
    if (data.kinect_image_url) {
        let kimg = document.getElementById("kinect_image");
        kimg.src = data.kinect_image_url;
        kimg.style.display = "block";
    } else {
        document.getElementById("kinect_image").style.display = "block";
    }

    // 显示 YOLO 检测图
    if (data.yolo_image_url) {
        let yimg = document.getElementById("yolo_detect_image");
        yimg.src = data.yolo_image_url;
        yimg.style.display = "block";
    } else {
        document.getElementById("yolo_detect_image").style.display = "block";
    }
}

//     // 图片显示逻辑
//     if (data.image_url) {
//         let img = document.getElementById("yolo_detect_image");
//         img.src = data.image_url;
//         img.style.display = "block";
//     } else {
//         document.getElementById("yolo_detect_image").style.display = "none";
//     }
// }

// 统一调用 API
function callApi(url, method = "POST", bodyData = {}, callback = null) {
    fetch(url, {
        method,
        headers: {'Content-Type': 'application/json'},
        body: method === "POST" ? JSON.stringify(bodyData) : null
    })
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.text(); // 先按文本获取
        })
        .then(text => {
            let data;
            try {
                data = JSON.parse(text); // 再尝试解析 JSON
            } catch (e) {
                throw new Error("返回数据不是 JSON: " + text.slice(0, 50));
            }
            updateUI(data);
            if (callback) callback(data);
        })
        .catch(err => {
            updateUI({status: "error", msg: "API 调用失败: " + err});
        });
}


// 初始化机械臂
function sendInit() {
    callApi("/robot/init");
}

// 机械臂复位
function sendReset() {
    callApi("/robot/reset");
}

// 移动机械臂
function sendMove() {
    callApi("/robot/move");
}


// Kinect 拍摄
function sendCaptureKinect() {
    callApi("/kinect/capture", "POST", {}, function (data) {
        if (data.status === "ok") {
            captureData.kinect = data;
            updateUI({status: "ok", msg: "Kinect 拍摄数据已保存"});
        } else {
            updateUI({status: "error", msg: "Kinect 拍摄失败"});
        }

    });
}

function sendCaptureRealsense() {
    callApi("/realsense/capture", "POST", {}, function (data) {
        if (data.status === "ok" && data.color_path && data.depth_path) {
            captureData.realsense = data;
            updateUI({status: "ok", msg: "Realsense 拍摄数据已保存"});
        } else {
            updateUI({status: "error", msg: "拍摄失败或数据缺失"});
        }
    });
}


// Kinect 检测
function sendDetectKinect() {
    if (!captureData.kinect) {
        updateUI({status: "error", msg: "请先拍摄 Kinect 数据"});
        return;
    }
    callApi("/kinect/yolo_detect", "POST", {kinect: captureData.kinect},
        function (data) {
            if (data.status === "ok" && data.detections) {
                captureData.yolo_results = data;
                updateUI({status: "ok", msg: "Realsense 目标检测数据已保存"});
            }
            else {
                updateUI({status: "error", msg: "Realsense 目标检测失败"});
            }
        });
}

//
// function sendRobotAutoMove() {
//     if (!captureData.yolo_results) {
//         updateUI({status: "error", msg: "请先对 Kinect 拍摄图像进行目标检测，并确保有有效检测结果"});
//         return;
//     }
//     // 2) 进行中提示
//     updateUI({status: "info", msg: "正在进行自动搬运任务..."});
//
//     callApi("/robot/auto_move", "POST", {
//         detections: captureData.yolo_results
//     }, function (data) {
//
//         if (data.status === "ok") {
//             updateUI({status: "ok", msg: data.msg || "自动搬运完成"});
//         } else {
//             updateUI({status: "error", msg: data.msg || "自动搬运失败"});
//         }
//         console.log(data.msg);
//     });
// }


function sendRobotAutoMove() {

    callApi("/robot/auto_move", "POST", {}, function (data) {

        if (data.status === "ok") {
            updateUI({status: "ok", msg: data.msg || "自动搬运完成"});
        } else {
            updateUI({status: "error", msg: data.msg || "自动搬运失败"});
        }
        console.log(data.msg);
    });
}

