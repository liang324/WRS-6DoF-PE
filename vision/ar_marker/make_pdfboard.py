import cv2
import numpy as np
from PIL import Image
import cv2.aruco as aruco

_MM_TO_INCH = 0.0393701


def drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length):
    """
    绘制 3D 坐标轴到图像上

    :param img: 要绘制坐标轴的图像
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param rvec: 旋转向量
    :param tvec: 平移向量
    :param length: 坐标轴的长度
    :return
    """
    axes_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # 绘制坐标轴
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X轴,蓝色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y轴,绿色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z轴,红色
    return img


def make_aruco_board(nrow,
                     ncolumn,
                     marker_dict=aruco.DICT_6X6_250,
                     start_id=0,
                     marker_size=25,
                     savepath='./',
                     name='test',
                     frame_size=(100, 100),
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """
    创建Aruco棋盘

    纸张为纵向,nrow表示垂直方向上的标记数量
    :param nrow: 行数
    :param ncolumn: 列数
    :param start_id: 标记的起始ID
    :param marker_dict: 标记字典
    :param marker_size: 标记大小
    :param savepath: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: (宽度, 高度) 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 无返回值

    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像
    markersizepx = int(marker_size * _MM_TO_INCH * dpi)  # 计算标记的像素大小
    markerdist = int(markersizepx / 4)  # 计算标记之间的距离

    if frame_size is not None:
        # 计算框架的像素大小并检查其是否适合纸张
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size[1] + 2:
            print("框架高度必须小于每列的#pt.")

        # 绘制框架
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0

    # 计算标记区域的大小和边距
    markerareanpxrow = (nrow - 1) * (markerdist) + nrow * markersizepx
    uppermargin = int((a4npxrow - markerareanpxrow) / 2)
    markerareanpxcolumn = (ncolumn - 1) * (markerdist) + ncolumn * markersizepx
    leftmargin = int((a4npxcolumn - markerareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return

    # 绘制每个标记
    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin + idnr * (markersizepx + markerdist)
            endrow = startrow + markersizepx
            startcolumn = leftmargin + idnc * (markersizepx + markerdist)
            endcolumn = markersizepx + startcolumn
            i = start_id + idnr * ncolumn + idnc
            img = aruco.drawDetectedMarkers(aruco_dict, i, markersizepx)
            bgimg[startrow:endrow, startcolumn:endcolumn] = img

    # 保存为PDF
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath + name + ".pdf", "PDF", resolution=dpi)


def make_charuco_board(nrow,
                       ncolumn,
                       marker_dict=aruco.DICT_4X4_250,
                       square_size=25,
                       save_path='./',
                       name='test',
                       frame_size=None,
                       paper_width=210,
                       paper_height=297,
                       dpi=600):
    """
    创建Charuco棋盘

    纸张为纵向,nrow表示垂直方向上的标记数量
    :param nrow: 行数
    :param ncolumn: 列数
    :param marker_dict: 标记字典
    :param square_size: 方块大小
    :param save_path: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: (宽度, 高度) 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 无返回值

    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    # 1毫米 = _MM_TO_INCH英寸
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像

    if frame_size is not None:
        # 计算框架的像素大小并检查其是否适合纸张
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size[1] + 2:
            print("框架高度必须小于每列的#pt.")

        # 绘制框架
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0

    # 计算方块区域的大小和边距
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx * nrow
    uppermargin = int((a4npxrow - squareareanpxrow) / 2)
    squareareanpxcolumn = squaresizepx * ncolumn
    leftmargin = int((a4npxcolumn - squareareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return

    # 创建Charuco棋盘
    # board = aruco.CharucoBoard_create(ncolumn, nrow, square_size, .57 * square_size, aruco_dict)
    board = aruco.CharucoBoard_create(ncolumn, nrow, square_size, .57 * square_size, aruco_dict)

    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin + squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin + squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard

    # 保存为PDF
    im = Image.fromarray(bgimg).convert("L")
    im.save(save_path + name + ".pdf", "PDF", resolution=dpi)


def make_dual_marker(marker_dict=aruco.DICT_4X4_250,
                     left_id=0,
                     right_id=1,
                     marker_size=45,
                     savepath='./',
                     name='test',
                     frame_size=(150, 60),
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """
    创建双标记 Aruco板

    纸张为纵向,nrow表示垂直方向上的标记数量
    :param left_id: 左侧标记的ID
    :param right_id: 右侧标记的ID
    :param marker_dict: 标记字典
    :param marker_size: 标记大小
    :param savepath: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: (宽度, 高度) 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 无返回值

    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.Dictionary_get(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像
    markersizepx = int(marker_size * _MM_TO_INCH * dpi)  # 计算标记的像素大小
    markerdist = int((frame_size[0] - (frame_size[1] - marker_size) - marker_size * 2) * _MM_TO_INCH * dpi)  # 计算标记之间的距离
    if frame_size is not None:
        frame_size_mm = [0.0, 0.0]
        frame_size_mm[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size_mm[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size_mm[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size_mm[1] + 2:
            print("框架高度必须小于每列的#pt.")
        framelft = int((a4npxcolumn - frame_size_mm[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size_mm[0])
        frametop = int((a4npxrow - frame_size_mm[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size_mm[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0
    markerareanpxrow = markersizepx
    uppermargin = int((a4npxrow - markerareanpxrow) / 2)
    markerareanpxcolumn = markerdist + markersizepx * 2
    leftmargin = int((a4npxcolumn - markerareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return
    for idnc in range(2):
        startrow = uppermargin
        endrow = startrow + markersizepx
        startcolumn = leftmargin + idnc * (markersizepx + markerdist)
        endcolumn = markersizepx + startcolumn
        i = left_id if idnc == 0 else right_id
        img = aruco.drawMarker(aruco_dict, i, markersizepx)
        bgimg[startrow:endrow, startcolumn:endcolumn] = img
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath + name + ".pdf", "PDF", resolution=dpi)


def make_multi_marker(marker_dict=aruco.DICT_4X4_250,
                      id_list=[0, 1, 2, 3, 4, 5],
                      marker_pos_list=((-80, 30), (-80, 0), (-80, -30), (80, 30), (80, 0), (80, -30)),
                      marker_size=25,
                      savepath='./',
                      name='test',
                      frame_size=(190, 130),
                      paper_width=210,
                      paper_height=297,
                      dpi=600):
    """
    创建多个标记的Aruco板

    纸张为纵向,nrow表示垂直方向上的标记数量
    0,0是纸张的中心,x指向右,y指向上
    :param id_list: 标记ID列表
    :param marker_pos_list: 标记位置列表
    :param marker_dict: 标记字典
    :param marker_size: 标记大小
    :param savepath: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: (宽度, 高度) 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 无返回值

    author: weiwei
    date: 20190420
    """
    if len(id_list) != len(marker_pos_list):
        print("错误:ID数量必须与标记位置数量相同！")
        exit(1)
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像
    markersizepx = int(marker_size * _MM_TO_INCH * dpi)  # 计算标记的像素大小
    # markerdist = int((frame_size[0] - (frame_size[1] - marker_size) - marker_size * 2) * _MM_TO_INCH * dpi)
    if frame_size is not None:
        frame_size_mm = [0.0, 0.0]
        frame_size_mm[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size_mm[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size_mm[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size_mm[1] + 2:
            print("框架高度必须小于每列的#pt.")
        framelft = int((a4npxcolumn - frame_size_mm[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size_mm[0])
        frametop = int((a4npxrow - frame_size_mm[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size_mm[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0
    for id_marker, pos_marker in zip(id_list, marker_pos_list):
        pos_marker_px = (
            a4npxrow / 2 - pos_marker[1] * _MM_TO_INCH * dpi, a4npxcolumn / 2 + pos_marker[0] * _MM_TO_INCH * dpi)
        start_row = int(pos_marker_px[0] - markersizepx / 2)
        end_row = int(pos_marker_px[0] + markersizepx / 2)
        start_column = int(pos_marker_px[1] - markersizepx / 2)
        end_column = int(pos_marker_px[1] + markersizepx / 2)
        img = aruco.drawDetectedMarkers(aruco_dict, id_marker, markersizepx)
        bgimg[start_row:end_row, start_column:end_column] = img
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath + name + ".pdf", "PDF", resolution=dpi)


def make_chess_board(nrow,
                     ncolumn,
                     square_size=25,
                     savepath='./',
                     name="test",
                     frame_size=None,
                     paper_width=210,
                     paper_height=297,
                     dpi=600):
    """
    创建棋盘格

    纸张为纵向,nrow表示垂直方向上的标记数量
    :param nrow: 行数
    :param ncolumn: 列数
    :param square_size: 方格大小,单位为毫米
    :param savepath: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: [宽度, 高度] 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 返回世界坐标点

    author: weiwei
    date: 20190420
    """
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size[1] + 2:
            print("框架高度必须小于每列的#pt.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)  # 计算方格的像素大小
    squareareanpxrow = squaresizepx * nrow
    uppermargin = int((a4npxrow - squareareanpxrow) / 2)
    squareareanpxcolumn = squaresizepx * ncolumn
    leftmargin = int((a4npxcolumn - squareareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return
    for idnr in range(nrow):
        for idnc in range(ncolumn):
            startrow = uppermargin + idnr * squaresizepx
            endrow = startrow + squaresizepx
            startcolumn = leftmargin + idnc * squaresizepx
            endcolumn = squaresizepx + startcolumn
            if idnr % 2 != 0 and idnc % 2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr % 2 == 0 and idnc % 2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
    im = Image.fromarray(bgimg).convert("L")
    im.save(savepath + name + ".pdf", "PDF", resolution=dpi)
    worldpoints = np.zeros((nrow * ncolumn, 3), np.float32)
    worldpoints[:, :2] = np.mgrid[:nrow, :ncolumn].T.reshape(-1, 2) * square_size
    return worldpoints


def make_chess_and_charuco_board(nrow_chess=3,
                                 ncolumn_chess=5,
                                 square_size=25,
                                 nrowch_aruco=3,
                                 ncolumn_charuco=5,
                                 marker_dict=aruco.DICT_6X6_250,
                                 square_size_aruco=25,
                                 save_path='./',
                                 name='test',
                                 frame_size=None,
                                 paper_width=210,
                                 paper_height=297,
                                 dpi=600):
    """
    创建半棋盘半Charuco板

    纸张为纵向,nrow表示垂直方向上的标记数量
    :param nrow_chess: 棋盘格行数
    :param ncolumn_chess: 棋盘格列数
    :param square_size: 棋盘格方格大小,单位为毫米
    :param nrowch_aruco: Charuco板行数
    :param ncolumn_charuco: Charuco板列数
    :param marker_dict: 标记字典
    :param square_size_aruco: Charuco板方格大小,单位为毫米
    :param save_path: 保存路径
    :param name: 保存的PDF文件名
    :param frame_size: (宽度, 高度) 1pt的框架用于方便裁剪,默认不绘制
    :param paper_width: 纸张宽度,单位为毫米
    :param paper_height: 纸张高度,单位为毫米
    :param dpi: 分辨率
    :return: 无返回值

    author: weiwei
    date: 20190420
    """
    aruco_dict = aruco.getPredefinedDictionary(marker_dict)
    a4npxrow = int(paper_height * _MM_TO_INCH * dpi)  # 计算A4纸张的像素行数
    a4npxcolumn = int(paper_width * _MM_TO_INCH * dpi)  # 计算A4纸张的像素列数
    bgimg = np.ones((a4npxrow, a4npxcolumn), dtype='uint8') * 255  # 创建白色背景图像
    if frame_size is not None:
        frame_size[0] = int(frame_size[0] * _MM_TO_INCH * dpi)
        frame_size[1] = int(frame_size[1] * _MM_TO_INCH * dpi)
        if a4npxcolumn < frame_size[0] + 2:
            print("框架宽度必须小于每行的#pt.")
        if a4npxrow < frame_size[1] + 2:
            print("框架高度必须小于每列的#pt.")
        framelft = int((a4npxcolumn - frame_size[0]) / 2 - 1)
        framergt = int(framelft + 1 + frame_size[0])
        frametop = int((a4npxrow - frame_size[1]) / 2 - 1)
        framedown = int(frametop + 1 + frame_size[1])
        bgimg[frametop:framedown + 1, framelft:framelft + 1] = 0
        bgimg[frametop:framedown + 1, framergt:framergt + 1] = 0
        bgimg[frametop:frametop + 1, framelft:framergt + 1] = 0
        bgimg[framedown:framedown + 1, framelft:framergt + 1] = 0

    # 上半部分,Charuco板
    squaresizepx = int(square_size_aruco * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx * nrow_chess
    uppermargin = int((a4npxrow / 2 - squareareanpxrow) / 2)
    squareareanpxcolumn = squaresizepx * ncolumn_chess
    leftmargin = int((a4npxcolumn - squareareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return
    board = aruco.CharucoBoard_create(ncolumn_chess, nrow_chess, square_size_aruco, .57 * square_size_aruco, aruco_dict)
    imboard = board.draw((squareareanpxcolumn, squareareanpxrow))
    print(imboard.shape)
    startrow = uppermargin
    endrow = uppermargin + squareareanpxrow
    startcolumn = leftmargin
    endcolumn = leftmargin + squareareanpxcolumn
    bgimg[startrow:endrow, startcolumn:endcolumn] = imboard

    # 下半部分,棋盘格
    squaresizepx = int(square_size * _MM_TO_INCH * dpi)
    squareareanpxrow = squaresizepx * nrowch_aruco
    uppermargin = int((a4npxrow / 2 - squareareanpxrow) / 2)
    squareareanpxcolumn = squaresizepx * ncolumn_charuco
    leftmargin = int((a4npxcolumn - squareareanpxcolumn) / 2)
    if (uppermargin <= 10) or (leftmargin <= 10):
        print("标记太多！减少行数和列数.")
        return
    for idnr in range(nrowch_aruco):
        for idnc in range(ncolumn_charuco):
            startrow = int(a4npxrow / 2) + uppermargin + idnr * squaresizepx
            endrow = startrow + squaresizepx
            startcolumn = leftmargin + idnc * squaresizepx
            endcolumn = squaresizepx + startcolumn
            if idnr % 2 != 0 and idnc % 2 != 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
            if idnr % 2 == 0 and idnc % 2 == 0:
                bgimg[startrow:endrow, startcolumn:endcolumn] = 0
    im = Image.fromarray(bgimg).convert("L")
    im.save(save_path + name + ".pdf", "PDF", resolution=dpi)


if __name__ == '__main__':
    # makechessandcharucoboard(4,6,32,5,7)
    # makecharucoboard(7,5, square_size=40)
    # makechessboard(7,5, square_size=40)
    # makearucoboard(2,2, marker_size=80)
    # make_aruco_board(1, 1, marker_dict=aruco.DICT_4X4_250, start_id=1, marker_size=45, frame_size=[60, 60])
    # print(type(aruco.Dictionary_get(aruco.DICT_4X4_250)))
    # result = aruco.drawMarker(dictionary=aruco.Dictionary_get(aruco.DICT_4X4_250), id=0, sidePixels=45)
    # print(type(result))
    # make_dual_marker(marker_dict=aruco.DICT_4X4_250, marker_size=45, dpi=600)
    # makechessboard(1, 1, square_size=35, frame_size = [100,150])

    make_multi_marker(marker_dict=aruco.DICT_4X4_250,
                      id_list=[0, 1, 2, 3, 4, 5, 6, 7],
                      marker_pos_list=(
                          (-77.5, 37.5), (-77.5, 12.5), (-77.5, -12.5), (-77.5, -37.5), (77.5, 37.5), (77.5, 12.5),
                          (77.5, -12.5), (77.5, -37.5)),
                      marker_size=20,
                      savepath='./',
                      name='test',
                      frame_size=(180, 130),
                      paper_width=210,
                      paper_height=297,
                      dpi=600)
