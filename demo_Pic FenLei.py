import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile

# 控制变量
runControl  = 4     # 运行模式：4-识别（鼠选）

PicDataGeShi= 4     # 数据格式：分类数据: 1-原始错图，6-画框错图，7-
                    #         原始数据: 2-分类存图，3-数据标定，4-分类画框，5-指定类别
PicDataType = 10    # 数据类型：9-CPC卡片数据，10-人脸HSJ数据，11-临时车牌数据，12-正式车牌数据, 13-LunHSJ数据, 14-UCAS数据

myConf = 0.25
myIou = 0.70
myBatchSize = 1

classIdx = 0

model = None        # 模型


def TypeDetect(dir_path):
    # 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 1 选择要进行测试的路径
    if len(dir_path) == 0:
        dir_path = filedialog.askdirectory()
        if dir_path is None:
            return

    # 2 设置保存路径
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if PicDataGeShi == 1:
        save_path2 = 'D:/识别错图pt_{:}_{:}'.format(PicDataType, curtime)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)

    if PicDataGeShi == 2:
        save_path2 = 'D:/原图分类pt_{:}'.format(PicDataType, curtime)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)

    if PicDataGeShi == 4 or PicDataGeShi == 5:
        save_path2 = 'D:/画框存图pt_{:}'.format(PicDataType, curtime)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)

    # 3 遍历文件夹下的所有图
    filelist = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('jpg') or file.endswith('bmp') or file.endswith('png'):
                filelist.append(os.path.join(root, file))
    files = sorted(filelist)
    if len(files) == 0:
        raise FileNotFoundError(f'{dir_path} does not exist')

    # 4 变量的初始化
    totalNum = len(files)
    curNum = 0
    curCount = 0
    errNum = 0
    t0 = time.time()
    imgs = []
    fileNames = []
    filePaths = []
    curPicNum = 0

    for filePath in reversed(files):

        # 5.1 取得文件名（不带路径）
        fileName = Path(filePath).name

        # 查错用代码: 程序崩溃时, 可以在再次运行时, 先将前面图片跳过,
        #if curCount < 118500:
        #    curCount = curCount + 1
        #    continue

        # print(fileName) # 打印文件名

        # 5.2 从文件夹上解析真实的类别名
        if PicDataGeShi == 1:
            realLabel = filePath[len(dir_path) + 1:len(dir_path) + 4]  # 二级目录文件夹值：3位数字
            if not realLabel.isdigit():  # 若无法转换为数字, 则使用二级目录的前6位字符
                realLabel = filePath[len(dir_path) + 1:len(dir_path) + 7]  # 二级目录文件夹值：6位标签

        # 5.3 原始文件无需解析
        if PicDataGeShi != 1:
            a = 1

        # 6 读取图片到im0中
        # 这样读图不安全，遇到没有ffd9结尾的图片，程序再后面会出错，故改成了后面的样式！！！
        # im0 = Image.open(filePath)
        with open(filePath, 'rb') as f:
            f = f.read()    # 这样读就是二进制的
        if len(f) < 1024:
            print(filePath) # 打印一下异常图的文件名
            continue
        f = f + B'\xff' + B'\xd9'   # 这句是补全数据的
        im0 = Image.open(BytesIO(f))
        if im0.mode != "RGB":
            im0 = im0.convert('RGB')

        if curPicNum == myBatchSize:
            imgs = []
            fileNames = []
            filePaths = []
            curPicNum = 0

        imgs.append(im0)
        fileNames.append(fileName)
        filePaths.append(filePath)

        curPicNum += 1
        if curPicNum < myBatchSize:
            continue

        # 7 进行推理预测
        # 不要传入图片路径，否则图片数量越多，占用的内存越多，2万张图能吃光64G内存
        #results = model.predict(source=im0, conf=myConf, iou=myIou)
        #results = model.predict(source=filePath, conf=myConf, iou=myIou, save=True)  # 保存画框图像
        #results = model.predict(source=filePath, conf=myConf, iou=myIou, save_txt=True)  # 保存识别结果
        results = model.predict(source=imgs, conf=myConf, iou=myIou)   # 多batch推理

        # 8 识别结果分析
        for i in range(myBatchSize): #对batch中的每一张图进行分析
            det = results[i].boxes
            curFileName = fileNames[i]

            recLabel = '0'
            zhidinglab = '0'
            maxConf = 0.0
            recFlag = 0
            for d in det:
                cls, conf = d.cls.squeeze(), d.conf.squeeze()

                if int(cls) == classIdx:
                    zhidinglab = model.names[int(cls)]

                if conf > maxConf:
                    recLabel = model.names[int(cls)]
                    maxConf = conf

                if PicDataType == 10:
                    if int(cls)==0:
                        recFlag += 1
                    if int(cls)==1:
                        recFlag += 1

            if PicDataType == 10:
                if recFlag == 2:
                    recLabel = 'facehsj'
                if recFlag >= 3:
                    recLabel = 'duojian'

            # 按指定的要求存图
            if PicDataGeShi == 1:  # 错误存图
                if realLabel != recLabel:
                    errNum = errNum + 1
                    nname = realLabel + '识别为' + recLabel
                    save_path1 = '{:}/{:}'.format(save_path2, nname)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = '{:}/{:}/{:}'.format(save_path2, nname, curFileName)
                    shutil.copyfile(filePath, save_path)

            if PicDataGeShi == 2:  # 分类存图
                save_path1 = '{:}/{:}'.format(save_path2, recLabel)
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)
                save_path = '{:}/{:}'.format(save_path1, curFileName)
                if maxConf < 1.55: # 需要权重过滤时
                    shutil.copyfile(filePath, save_path)
                    errNum = errNum + 1
                #shutil.move(filePath, save_path)

            if PicDataGeShi == 3:   # 数据标定
                save_path = filePaths[i].replace('jpg', 'txt')
                f = open(save_path, 'a')
                boxes = results[i].boxes
                for box in boxes:
                    cat_num = int(box.cls.cpu())
                    label = box.xywhn.cpu().numpy()
                    size = label[0].tolist()
                    size_string = ' '.join(map(str, size))
                    result = f'{cat_num} {size_string}\n'
                    print('result', result)
                    f.write(str(result))
                f.close()

            if PicDataGeShi == 4:   # 分类画框存图
                im_array = results[i].plot()  # 绘制预测结果的BGR numpy数组
                img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
                save_path1 = '{:}/{:}'.format(save_path2, recLabel)
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)
                save_path = '{:}/{:}/{:}'.format(save_path2, recLabel, curFileName)
                cv2.imwrite(save_path, img)
                errNum = errNum + 1

            if PicDataGeShi == 5 and zhidinglab != '0':   # 指定类画框存图
                im_array = results[i].plot()  # 绘制预测结果的BGR numpy数组
                img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
                save_path1 = '{:}/{:}'.format(save_path2, zhidinglab)
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)
                save_path = '{:}/{:}/{:}'.format(save_path2, zhidinglab, curFileName)
                cv2.imwrite(save_path, img)
                errNum = errNum + 1

        # 15 每隔100张图片打印一次识别率, 既可以看进度，也可以判定是否还要继续运行
        curNum = curNum + myBatchSize
        curCount = curCount + myBatchSize
        if curNum >= 100:
            curNum = 0

            t = int((time.time() - t0) * (totalNum - curCount) / curCount)
            h = t // 3600
            m = (t % 3600) // 60
            s = t % 60
            t1 = int(time.time() - t0)
            h1 = t1 // 3600
            m1 = (t1 % 3600) // 60
            s1 = t1 % 60

            print(f'{curCount}/{totalNum}，总错{errNum}，总率{(curCount - errNum) / curCount:.2%}   '
                      f'已耗时间：{h1}时{m1}分{s1}秒，剩余时间：{h}时{m}分{s}秒\n')

    # 16 识别率计算、显示并写入文件中
    if len(save_path2) > 0:
        save_path = '{:}/{:}'.format(save_path2, '识别率.txt')
        file = open(save_path, 'w')
        tmpStr = '总错{:}，总数{:}，总率{:.2%}，总耗时：{:.2f}s\n'.format(
            errNum, totalNum, (totalNum - errNum) / totalNum, time.time() - t0)
        print(tmpStr)
        file.write(tmpStr)
        file.close()




if __name__ == "__main__":
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    saveName = "train{:}".format(curtime)

    # 1 加载模型
    model_path = filedialog.askopenfilename()
    if model_path is None:
        print("请选择一个模型文件(pt)，否则无法进行推理！")
    else:
        model = YOLO(model_path)

    # 推理1：通过对话框选择样本路径，逐个识别并统计识别率
    if runControl == 4:
        TypeDetect("")



    # 特别说明：以下的代码只使用单个模型，不使用组合模型

    # 推理3：指定目录，画框存图
    if runControl == 6:
        path = filedialog.askdirectory()  # 通过文件夹选择要进行测试的路径
        results = model.predict(source=path, save=True) # 展示预测结果

    im1 = Image.open("D:/11.jpg")
    im2 = cv2.imread("E:/DLDataSets/bus.jpg")

    # 推理4：检测图片
    if runControl == 7:
        results = model("D:/11.jpg")
        res = results[0].plot()
        cv2.imshow("YOLOv8 Inference", res)
        cv2.waitKey(0)

    if runControl == 11:  # from PIL
        results = model.predict(source=im1, save=True)  # 保存绘制的图像

    if runControl == 12:  # from ndarray
        results = model.predict(source=im2, save_txt=True)  # 将预测保存为标签

    if runControl == 13:  # from list of PIL/ndarray
        results = model.predict(source=[im1, im2])



# 如何让YOLO的推理得更快，总体看来，主要有以下这些思路：
#     使用更快的 GPU，即：P100 -> V100 -> A100
#     多卡GPU推理
#     减小模型尺寸，即YOLOv5x -> YOLOv5l -> YOLOv5m -> YOLOv5s -> YOLOv5n
#     进行半精度FP16推理 python detect.py --half
#     减少–img-size，即 1280 -> 640 -> 320
#     导出成ONNX或OpenVINO格式，获得CPU加速
#     导出到TensorRT获得GPU加速
#     批量输入图片进行推理
#     使用多进程/多线程进行推理





