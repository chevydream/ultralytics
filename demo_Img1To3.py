import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile

def Img1To3():
    # 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 1 选择要进行分割的图片及标签路径
    dir_path = filedialog.askdirectory()
    if dir_path is None:
        return
    lab_path = filedialog.askdirectory()
    if lab_path is None:
        return

    # 2 创建存储分割图片和标签的文件夹
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Img3path = dir_path + curtime
    os.mkdir(Img3path)
    lab2path = lab_path + curtime
    os.mkdir(Img3path)

    # 3 遍历文件夹下的所有图
    filelist = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('jpg') or file.endswith('bmp') or file.endswith('png'):
                filelist.append(os.path.join(root, file))
    files = sorted(filelist)
    if len(files) == 0:
        raise FileNotFoundError(f'{dir_path} does not exist')

    totalNum = len(files)
    iouErrNum = 0
    iouCurNum = 0
    for filePath in reversed(files):

        iouCurNum = iouCurNum + 1

        # 4 取得文件名（不带路径）
        fileName = Path(filePath).name
        # print(fileName) # 打印文件名

        # 5 从标签中读取类别和坐标
        label_path = '{:}/{:}txt'.format(lab_path, fileName[0:len(fileName)-3])
        if os.path.exists(label_path):
            boxes_info = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # 标签文件中class_labels的数量
            realboxes_len = boxes_info.size()[0]
        else:
            realboxes_len = 0

        # 6 读取图片到im0中
        # im0 = Image.open(filePath) # 不安全，遇到没ffd9结尾的图，程序后面会出错，故改成了后面的样式！！！
        with open(filePath, 'rb') as f:
            f = f.read()    # 这样读就是二进制的
        if len(f) < 1024:
            print(filePath) # 打印一下异常图的文件名
            continue
        f = f + B'\xff' + B'\xd9'   # 这句是补全数据的
        im0 = Image.open(BytesIO(f))
        if im0.mode != "RGB":
            im0 = im0.convert('RGB')

        # 7 分割成3张图
        if im0.width == 1920:
            CutImg1 = im0[0:im0.height, 0:640]
            CutImg2 = im0[0:im0.height, 640:640]
            CutImg3 = im0[0:im0.height, 1280:im0.width-1280]
        if im0.width == 1280:
            CutImg1 = im0[0:im0.height, 0:432]
            CutImg2 = im0[0:im0.height, 432:432]
            CutImg3 = im0[0:im0.height, 864:im0.width-864]

        # 8 标签分成3份
        for i=0, realboxes_len
        boxes_info

        # 9 存图
        save_ImgPath = '{:}/{:}'.format(save_path2, fileName)
        save_LabPath = '{:}/{:}txt'.format(save_path2, fileName[0:len(fileName) - 3])

        # 9.2 提取标定和检测不一样的画框图(检测框)
        if iouErrFlag == 1 and runControl == 2:
            for r in results:
                im_array = r.plot()  # 绘制预测结果的BGR numpy数组
                img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
            # cv2_imshow(img)  # 显示图像
            cv2.imwrite(save_ImgPath, img)

        # 10 打印进度
        print(f'进度：{iouCurNum}/{totalNum}, 错{iouErrNum}, '
              f'率{iouCurNum / (iouErrNum+iouCurNum):.2%}  ' + fileName)



if __name__ == "__main__":

    Img1To3()




