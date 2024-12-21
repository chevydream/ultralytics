import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile
import tkinter.messagebox
from demo_CommFun import bbox_iou_xywh, bbox_iou_xywhn

# 控制变量
runControl  = 4  # 运行模式：1-存原始图, 2-存画框图(检测框), 3-存画框图(标签框), 4-存画框图(差异框)，5-追加标定

dataControl = 1  # 数据类型: 1-通用数据, 2-G3车型图, 3-G3头尾图, 4-UpDown图

codeControl = 3  # 代码控制: 1-鼠选模型, 2-鼠选样本, 3-全部鼠选, 4-全部默认
modelPath   = "E:\\DLRuns\\detectV8\\train20240714_071615\\weights\\best.pt"
imgPath     = "E:\\DLDataSets\\UpDown\\images\\train"
labPath     = "E:\\DLDataSets\\UpDown\\labels\\train"

myConf = 0.25
myIou = 0.70

model = None  # 单个模型

nd_fName_G3 = {
    'k1gw': 0, 'k120': 1, 'k220': 2, 'k320': 3, 'k420': 4, 'km20': 5,
    'k2fc': 2, 'k219': 2, 'k2gc': 2, 'k2tg': 2, 'dc20': 98,
    'jtd2': 9, 'hm20': 10, 'h120': 11, 'h220': 12, 'h320': 13, 'h420': 14, 'h520': 15, 'h620': 16,
    'h2gw': 17, 'hN20': 19, 'z220': 22, 'z320': 23, 'z420': 24, 'z520': 25, 'z620': 26,
    'mt20': 38, 'fgzz': 40, 'fgcc': 41, 'xs20': 42, 'ds20': 43, 'rs20': 44, 'fc20': 99,
    'carH': 0, 'moto': 1}

nd_fName_CarHead = {'car20': 0, 'motor': 1, 'truck': 2}

nd_fName_UpDown = {'face': 0, 'hsjDown': 1, 'hsjUp': 2, 'hand': 3}

def TypeDetect():

    # 1 准备工作

    # 1.0 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 1.1 选择要进行结果分析的路径
    if codeControl == 2 or codeControl == 3:
        dir_path = filedialog.askdirectory()
        if dir_path is None:
            return
        lab_path = filedialog.askdirectory()
        if lab_path is None:
            return
    else:
        dir_path = imgPath
        if dir_path is None:
            return
        lab_path = labPath
        if lab_path is None:
            return

    # 1.2 创建存储错图的文件夹
    name = ["未知", "识错", "多检", "漏检", "双类异", "双类同", "背景", "多检wxp", "漏检wxp", "多检llc", "漏检llc", "多检双", "漏检双"]
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path2 = 'D:/目标标定iou_{:}'.format(curtime)
    if not os.path.exists(save_path2):
        os.mkdir(save_path2)
        save_path = '{:}/{:}/'.format(save_path2, "标错")
        os.mkdir(save_path)
        for i in range(len(name)):
            save_path = '{:}/{:}/'.format(save_path2, name[i])
            os.mkdir(save_path)

    # 1.3 遍历文件夹下的所有图
    filelist = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('jpg') or file.endswith('bmp') or file.endswith('png'):
                filelist.append(os.path.join(root, file))
    files = sorted(filelist)
    if len(files) == 0:
        raise FileNotFoundError(f'{dir_path} does not exist')

    totalNum = len(files)
    iouErrNum = [0, 0, 0, 0, 0, 0, 0]
    iouCurNum = 0
    for filePath in reversed(files):

        iouCurNum = iouCurNum + 1

        # 2 单张的前准备

        # 2.1 跳过开头的N张图, 查错时使用
        # if iouCurNum < 10440:
        #    continue

        # 2.2 取得文件名（不带路径）
        fileName = Path(filePath).name
        # print(fileName) # 打印文件名

        # 3 读取标定信息(含: 查错, 去多余, 提双类, 提wxp和llc)
        label_path = '{:}/{:}txt'.format(lab_path, fileName[0:len(fileName) - 3])
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:

            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # 3.1 标签中是否有文件名上的类别, 提取标定错误的样本
            cls_info = boxes[:, 0]
            findFlag = 0
            for i in range(len(cls_info)):
                if dataControl == 1:
                    findFlag = 1
                if dataControl == 2:
                    if cls_info[i] == nd_fName_G3[fileName[0:4]] \
                            or nd_fName_G3[fileName[0:4]] == 98 or nd_fName_G3[fileName[0:4]] == 99:
                        findFlag = 1
                if dataControl == 3:
                    findFlag = 1
                if dataControl == 4:
                    findFlag = 1
            if findFlag == 0:
                save_ImgPath = '{:}/{:}/{:}'.format(save_path2, "标错", fileName)
                save_LabPath = '{:}/{:}/{:}txt'.format(save_path2, "标错", fileName[0:len(fileName) - 3])
                shutil.copyfile(filePath, save_ImgPath)
                shutil.copyfile(label_path, save_LabPath)

            # 3.2 提取特定的标签(具有双标签的检测框)
            doubleCount = 0
            dBoxes = torch.zeros(len(boxes), 5)
            for box in boxes:
                if (dataControl == 1 and (box[0] == 100 or box[0] == 101)) \
                        or (dataControl == 2 and (box[0] == 0 or box[0] == 1 or box[0] == 5 or box[0] == 6)) \
                        or (dataControl == 3 and (box[0] == 100 or box[0] == 101))\
                        or (dataControl == 4 and (box[0] == 100 or box[0] == 101)):
                    dBoxes[doubleCount] = box
                    doubleCount = doubleCount + 1
            if doubleCount > 0:
                doubleBoxes = dBoxes[0:doubleCount]

            # 3.3 去除特定的标签(训练时没有参与训练的标签, 双标签)
            if dataControl == 2:
                for a in [0, 1, 5, 6, 30, 31, 40, 41, 42, 43, 44]:
                    boxes = boxes[boxes[:, 0] != a]

            # 3.4 获取类别, 坐标和数量
            clses_info = boxes[:, 0]
            boxes_info = boxes[:, 1:]
            realboxes_len = boxes_info.size()[0]

            if (realboxes_len != 0):
                boxes_info = boxes_info.cpu().numpy()
                clses_info = clses_info.cpu().numpy()
                clses_bak = clses_info.copy()

            if doubleCount > 0:
                doubleClses_info = doubleBoxes[:, 0]
                doubleBoxes_info = doubleBoxes[:, 1:]
                doubleboxes_len = doubleBoxes_info.size()[0]
            else:
                doubleboxes_len = 0

            # 3.5 找出wxp和llc样本
            wxpllc_real = 0
            if dataControl == 2 and realboxes_len > 1:
                for i in range(0, realboxes_len):
                    if clses_info[i] == 28:  # 识别为wxp
                        wxpllc_real = 2
                        clses_bak[i] = -1
                    if clses_info[i] == 29:  # 识别为llc
                        wxpllc_real = 3
                        clses_bak[i] = -1

            # 3.6 判断双标签是否有错误
            if doubleboxes_len % 2 != 0:
                tkinter.messagebox.askokcancel("提示", "{:}出现了双标签错误!".format(fileName))
        else:
            realboxes_len = 0
            doubleboxes_len = 0
            wxpllc_real = 0

        # 4 读取图片到im0中
        # im0 = Image.open(filePath) # 不安全，遇到没ffd9结尾的图，程序后面会出错，故改成了后面的样式！！！
        with open(filePath, 'rb') as f:
            f = f.read()  # 这样读就是二进制的
        if len(f) < 1024:
            print(filePath)  # 打印一下异常图的文件名
            continue
        f = f + B'\xff' + B'\xd9'  # 这句是补全数据的
        im0 = Image.open(BytesIO(f))
        if im0.mode != "RGB":
            im0 = im0.convert('RGB')

        # 5 进行推理预测
        # 不要传入图片路径，否则图片数量越多，占用的内存越多，2万张图能吃光64G内存
        results = model.predict(source=im0, conf=myConf, iou=myIou)

        # 6 识别结果分析(找出:多检的,漏检的和错判的,存图)
        # iouErrFlag的取值范围: 0:未知, 1:识错, 2:多检, 3:漏检, 4:双类, 5:正确, 6:背景
        iouErrFlag = [0, 0, 0, 0, 0, 0, 0]
        for rst in reversed(results):  # 当BatchSize不是1时, results就是多个了
            wxpllc_Detect = 0
            dJianChuNum = 0
            recboxes_len = rst.boxes.xywhn.size()[0]

            # 6.1 找出被检测成双类的样本, 进行标记
            doubleCls = []
            doubleCls.append(0)
            if recboxes_len > 1:
                for i in range(1, recboxes_len):
                    doubleCls.append(0)
                    for j in range(i):
                        boxi = rst.boxes[i].xywhn.cpu().numpy()
                        boxj = rst.boxes[j].xywhn.cpu().numpy()
                        clsi = int(rst.boxes[i].cls.cpu())
                        clsj = int(rst.boxes[j].cls.cpu())
                        iou = bbox_iou_xywh(boxi.squeeze(), boxj.squeeze())
                        if iou > 0.85:
                            flag1 = dataControl == 1 and ((clsi == 100 and clsj == 101) or (clsj == 100 or clsi == 101))
                            flag2 = dataControl == 2 and ((clsi == 1 and (clsj == 0 or clsj == 5 or clsj == 6)) \
                                    or (clsj == 1 and (clsi == 0 or clsi == 5 or clsi == 6)) \
                                    or ((clsi == 0 or clsi == 5 or clsi == 6) and (clsj == 0 or clsj == 5 or clsj == 6)))
                            flag3 = dataControl == 3 and ((clsi == 100 and clsj == 101) or (clsj == 100 or clsi == 101))
                            if clsi == clsj:
                                iouErrFlag[4] = 2   # 双类同
                                if rst.boxes[i].conf.squeeze() < rst.boxes[j].conf.squeeze():
                                    doubleCls[i] = 1
                                else:
                                    doubleCls[j] = 1
                            elif flag1 or flag2 or flag3:
                                if rst.boxes[i].conf.squeeze() < rst.boxes[j].conf.squeeze():
                                    doubleCls[i] += 2
                                else:
                                    doubleCls[j] += 2
                            else:
                                iouErrFlag[4] = 1  # 双类异
                                if rst.boxes[i].conf.squeeze() < rst.boxes[j].conf.squeeze():
                                    doubleCls[i] = 1
                                else:
                                    doubleCls[j] = 1

            for i in range(recboxes_len):
                box = rst.boxes[i]
                det_cls = int(box.cls.cpu())
                det_box = box.xywhn.cpu().numpy()

                # 6.2 找出wxp和llc
                if dataControl == 2:
                    if int(rst.boxes[i].cls.cpu()) == 28:  # 识别为wxp
                        wxpllc_Detect = 2
                    if int(rst.boxes[i].cls.cpu()) == 29:  # 识别为llc
                        wxpllc_Detect = 3

                # 6.3 计算iou(因为两个标签是同一个框,故必有两个重叠框满足条件)
                sumIou = sameCls = 0
                if doubleboxes_len > 0:
                    diou = [0, 0, 0, 0, 0, 0, 0, 0]
                    for j in range(0, doubleboxes_len, 2):
                        diou[j + 0] = 1 if bbox_iou_xywh(det_box.squeeze(), doubleBoxes_info[j + 0]) > 0.75 else 0
                        diou[j + 1] = 1 if bbox_iou_xywh(det_box.squeeze(), doubleBoxes_info[j + 1]) > 0.75 else 0
                        sumIou = sumIou + diou[j + 0] + diou[j + 1]
                        if (diou[j + 0] == 1 and doubleClses_info[j + 0] == det_cls):
                            sameCls += 1
                        if (diou[j + 1] == 1 and doubleClses_info[j + 1] == det_cls):
                            sameCls += 1

                # 6.4 当前检测框为双类类别时的结果分析
                if sumIou > 0:

                    dJianChuNum += 1

                    # 双类时, 只分析权重高的那个框, 低的跳过
                    if doubleCls[i] == 4:
                        continue

                    if sumIou == 2:
                        if sameCls == 1:
                            iouErrFlag[5] = 1  # 正确
                            iouErrNum[5] += 1
                        else:
                            iouErrFlag[1] = 1  # 识错
                            iouErrNum[1] += 1
                    else:
                        save_ImgPath = '{:}/{:}/{:}'.format(save_path2, "标错", fileName)
                        save_LabPath = '{:}/{:}/{:}txt'.format(save_path2, "标错", fileName[0:len(fileName) - 3])
                        shutil.copyfile(filePath, save_ImgPath)
                        shutil.copyfile(label_path, save_LabPath)

                # 6.5 当前检测框为非双类类别时的结果分析
                else:

                    # 双类时, 只分析权重高的那个框, 低的跳过
                    if doubleCls[i] == 1:
                        continue

                    # wxp和llc单独分析, 不在这里处理, 跳过
                    if dataControl == 2 and (det_cls == 28 or det_cls == 29):
                        continue

                    # 计算iou, 寻找重叠度最大的目标
                    maxIoU = 0
                    maxi = -1
                    for j in range(realboxes_len):
                        iou = bbox_iou_xywh(det_box.squeeze(), boxes_info[j])
                        if maxIoU < iou:
                            maxIoU = iou
                            maxi = j

                    # 这个阈值很关键, 太大/太小都不合适, 调时要慎重
                    a1 = (dataControl == 2 and (maxIoU > 0.7 or (realboxes_len == 1 and maxIoU > 0.5)))
                    a2 = (dataControl != 2 and maxIoU > 0.5)
                    if a1 or a2:
                        clses_bak[maxi] = -1
                        if clses_info[maxi] == det_cls:
                            iouErrFlag[5] = 1  # 正确
                            iouErrNum[5] += 1
                        else:
                            iouErrFlag[1] = 1  # 识错
                            iouErrNum[1] += 1
                    else:
                        iouErrFlag[2] = 1  # 多检
                        iouErrNum[2] += 1

            # 6.6 结果分析的剩余工作
            for i in range(realboxes_len):
                if clses_bak[i] != -1:
                    iouErrFlag[3] = 1  # 漏检
                    iouErrNum[3] += 1

            if realboxes_len == 0 and recboxes_len == 0:
                iouErrFlag[6] = 1  # 背景
                iouErrNum[6] += 1

            # wxp和llc的漏检和多检
            if dataControl == 2 and wxpllc_real != wxpllc_Detect:
                if wxpllc_real == 2 and wxpllc_Detect == 0:
                    iouErrFlag[3] = 2  # 漏检(wxp)
                    # iouErrNum[3] += 1
                if wxpllc_real == 3 and wxpllc_Detect == 0:
                    iouErrFlag[3] = 3  # 漏检(llc)
                    # iouErrNum[3] += 1
                if wxpllc_real == 0 and wxpllc_Detect == 2:
                    iouErrFlag[2] = 2  # 多检(wxp)
                    # iouErrNum[2] += 1
                if wxpllc_real == 0 and wxpllc_Detect == 3:
                    iouErrFlag[2] = 3  # 多检(llc)
                    # iouErrNum[2] += 1

            if dJianChuNum < doubleboxes_len:
                iouErrFlag[3] = 4  # 漏检双
                iouErrNum[3] += 1
            if dJianChuNum > doubleboxes_len:
                iouErrFlag[2] = 4  # 多检双
                iouErrNum[2] += 1

            if iouErrFlag[1] == 0 and iouErrFlag[2] == 0 and iouErrFlag[3] == 0 \
                    and iouErrFlag[4] == 0 and iouErrFlag[5] == 0 and iouErrFlag[6] == 0:
                iouErrFlag[0] = 1  # 未知情况
                iouErrNum[0] += 1

        # 7 按指定的要求存图
        for i in range(5):  # iouErrFlag的取值范围: 0:未知, 1:识错, 2:多检, 3:漏检, 4:双类
            if iouErrFlag[i] > 0:
                # 7.0 构建图片和标签的目的路径
                if iouErrFlag[i] == 1:
                    save_ImgPath = '{:}/{:}/{:}'.format(save_path2, name[i], fileName)
                    save_LabPath = '{:}/{:}/{:}txt'.format(save_path2, name[i], fileName[0:len(fileName) - 3])
                elif i == 4 and iouErrFlag[i] == 2:
                    save_ImgPath = '{:}/{:}/{:}'.format(save_path2, name[5], fileName)
                    save_LabPath = '{:}/{:}/{:}txt'.format(save_path2, name[5], fileName[0:len(fileName) - 3])
                else:
                    save_ImgPath = '{:}/{:}/{:}'.format(save_path2, name[i + iouErrFlag[i] * 2 + 1], fileName)
                    save_LabPath = '{:}/{:}/{:}txt'.format(save_path2, name[i + iouErrFlag[i] * 2 + 1], fileName[0:len(fileName) - 3])

                # 7.1 提取标定和检测不一样的原始数据
                if runControl == 1:
                    shutil.copyfile(filePath, save_ImgPath)
                    if realboxes_len != 0:
                        shutil.copyfile(label_path, save_LabPath)

                # 7.2 提取标定和检测不一样的画框图(检测框)
                if runControl == 2:
                    for r in results:
                        im_array = r.plot()  # 绘制预测结果的BGR numpy数组
                        img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
                    # cv2_imshow(img)  # 显示图像
                    cv2.imwrite(save_ImgPath, img)

                # 7.3 提取标定和检测不一样的画框图(标签框)
                if runControl == 3:
                    width = im0.width
                    height = im0.height
                    if (realboxes_len != 0):
                        for r in boxes_info:
                            first_point = (int(r[0] * width - r[2] * width / 2), int(r[1] * height - r[3] * height / 2))
                            last_point = (int(r[0] * width + r[2] * width / 2), int(r[1] * height + r[3] * height / 2))
                            cv2.rectangle(im0, first_point, last_point, (0, 255, 0), 2)
                    cv2.imwrite(save_ImgPath, im0)

                # 7.4 提取标定和检测不一样的画框图(差异框)
                if runControl == 4:  #
                    for r in results:
                        im_array = r.plot()  # 绘制预测结果的BGR numpy数组
                        img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)

                    width = im0.width
                    height = im0.height
                    if (realboxes_len != 0):
                        for r in boxes_info:
                            center = (int(r[0] * width), int(r[1] * height))
                            cv2.ellipse(img, center, (int(r[2] * width / 2), int(r[3] * height / 2)), 0, 0, 360, (0, 255, 0), 2)
                    if (doubleboxes_len != 0):
                        for r in doubleBoxes_info:
                            center = (int(r[0] * width), int(r[1] * height))
                            cv2.ellipse(img, center, (int(r[2] * width / 2), int(r[3] * height / 2)), 0, 0, 360, (0, 255, 0), 2)

                    # cv2_imshow(img)  # 显示图像
                    cv2.imwrite(save_ImgPath, img)

        # 8 给标定文件追加新检测出来的目标
        if runControl == 5:

            # 8.1 读取原内容
            with open(label_path, 'r') as f:
                data = f.readlines()

            with open(save_LabPath, 'w') as f:
                # 8.2 逐行写入原内容
                for line in data:
                    f.write(line)

                # 8.3 追加新目标写入
                boxes = results[0].boxes
                for box in boxes:
                    cat_num = int(box.cls.cpu())
                    if cat_num == 1 or cat_num == 3:
                        label = box.xywhn.cpu().numpy()
                        size = label[0].tolist()
                        size_string = ' '.join(map(str, size))
                        result = f'{cat_num} {size_string}\n'
                        f.write(str(result))

        # 9 打印进度
        if iouCurNum % 10 == 0:
            cuoNum = iouErrNum[0] + iouErrNum[1] + iouErrNum[2] + iouErrNum[3]
            duiNum = iouErrNum[5] + iouErrNum[6]
            print(f'进度：{iouCurNum}/{totalNum}, 对{duiNum}={iouErrNum[5]}+{iouErrNum[6]}, '
                  f'错{cuoNum}={iouErrNum[0]}+{iouErrNum[1]}+{iouErrNum[2]}+{iouErrNum[3]}, '
                  f'率{duiNum / (cuoNum+duiNum):.2%}  ' + fileName)


if __name__ == "__main__":
    # 1 加载模型
    if codeControl == 1 or codeControl == 3:
        model_path = filedialog.askopenfilename()
        if model_path is None:
            print("请选择一个模型文件(pt)，否则无法进行推理！")
        else:
            model = YOLO(model_path)
    else:
        model = YOLO(modelPath)

    # 2 通过对话框选择样本路径，逐个识别，存错误图
    TypeDetect()
