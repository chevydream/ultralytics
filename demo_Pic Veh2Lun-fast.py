import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile
from demo_CommFun import bbox_iou_xywh, bbox_iou_xywhn, cal_iou

# 控制变量
modelSelect = 4     # 模型版本：0-未定义，1-鼠选模型，2-v8模型，3-v10模型，4-v11模型
recTypeFlag = 2     # 模型类型：0-未定义, 1-未定义，2-车型识别，3-车轮检测，4-轮+hsj

runControl  = 4     # 数据来源：4-鼠选选路径，5-批量跑路径
PicDataGeShi= 1     # 数据格式：0-标定数据（存IOU图），1-分类数据（只存错图），2-原始数据（分类存图），3-单类存错（只存错图）
PicDataType = 1     # 数据内容：1-车道车型数据，2-车道HSJ轮数据，3-车道危险数据，4-车道冷链数据
                    #         5-门架车型数据，6-门架集装箱数据，7-门架危险数据，8-门架冷链数据

FLBZFlag    = 1        # 1-国标, 2-墨西哥分类标准

myConf      = 0.25
myIou       = 0.70

myBatchSize = 1

model = None        # 单个模型

# Yolo模型标签与国标的对应关系
yolo_type_dict_gb = dict(car1=1, car2=2, car3=3, car4=4, keMian=1, car1gw3=1, car1gw4=1,
                    truck1=11, truck2=12, truck3=13, truck4=14, truck5=15, truck6=16,
                    huoMian=11, truckN=16, trk2T1A=12, JianTou2=12, trk2gw3=12, trk2gw1=12,
                    zhuan1=21, zhuan2=22, zhuan3=23, zhuan4=24, zhuan5=25, zhuan6=26,
                    lun=31, wxp=32, llc=33, hsj=34, k1ZiFu=41, k2ZiFu=42, k3ZiFu=43, k4ZiFu=44,
                    motor=0, tuolaji=0, chanche=0, chache=0, XiaoSanlun=0, DaSanlun=0, RenSanlun=0,
                    carHead=0, motorHead=0, carTail=0, motorTail=0)

# Yolo模型标签与墨西哥标准的对应关系
yolo_type_dict_mxg = dict(car1=2, car2=2, car3=3, car4=3, keMian=2, car1gw3=2, car1gw4=2,
                    truck1=2, truck2=6, truck3=7, truck4=8, truck5=9, truck6=10,
                    huoMian=2, truckN=11, trk2T1A=2, JianTou2=6, trk2gw3=2, trk2gw1=6,
                    zhuan1=2, zhuan2=6, zhuan3=7, zhuan4=8, zhuan5=9, zhuan6=10,
                    lun=31, wxp=32, llc=33, hsj=34, k1ZiFu=41, k2ZiFu=42, k3ZiFu=43, k4ZiFu=44,
                    motor=1, tuolaji=0, chanche=0, chache=0, XiaoSanlun=0, DaSanlun=0, RenSanlun=0,
                    carHead=0, motorHead=0, carTail=0, motorTail=0)

# 文件夹名称与国标的对应关系
dirname_type_dict_gb = {'101': 1, '202': 2, '301': 3, '302': 4, '501': 11, '702': 12, '711': 22, '801': 13,
                   '809': 23, '901': 14, '915': 24, '953': 15, '970': 25, '979': 16, '997': 26,
                   'carke1': 1, 'carke2': 2, 'carke3': 3, 'carke4': 4, 'truckN': 16,
                   'truck1': 11, 'zhuan1': 21, 'truck2': 12, 'zhuan2': 22, 'truck3': 13, 'zhuan3': 23,
                   'truck4': 14, 'zhuan4': 24, 'truck5': 15, 'zhuan5': 25, 'truck6': 16, 'zhuan6': 26,
                   'feiche': 0, 'feigao': 0, 'chanch': 0, 'chache': 0, 'Xsanlu': 0, 'Rsanlu': 0, 'Dsanlu': 0}

# 文件夹名称与墨西哥标准的对应关系
dirname_type_dict_mxg = {'001': 1, '002': 2, '003': 3, '004': 4, '005': 5, '006': 6, '007': 7, '008': 8,
                         '009': 9, '010': 10, '011': 11, '012': 12, '013': 13,
                         'TM': 1, 'T1A': 2, 'TB2': 3, 'TB3': 4, 'TB4': 5, 'T2C': 6, 'T3C': 7, 'T4C': 8,
                         'T5': 9, 'T6': 10, 'T7': 11, 'T8': 12, 'T9': 13}

# 文件夹名称与轴数的对应关系
dirname_lun_dict = {'101': 2, '202': 2, '301': 2, '302': 2, '501': 2, '702': 2, '711': 2, '801': 3,
                   '809': 3, '901': 4, '915': 4, '953': 5, '970': 5, '979': 6, '997': 6,
                   'feiche': 0, 'carke1': 2, 'carke2': 2, 'carke3': 2, 'carke4': 2, 'truckN': 10,
                   'truck1': 2, 'zhuan1': 2, 'truck2': 2, 'zhuan2': 2, 'truck3': 3, 'zhuan3': 3,
                   'truck4': 4, 'zhuan4': 4, 'truck5': 5, 'zhuan5': 5, 'truck6': 6, 'zhuan6': 6,
                   'feigao': 2, 'chanche': 2, 'chache': 2, 'Xsanlun': 2, 'Rsanlun': 2, 'Dsanlun': 2,
                   '001': 2, '002': 2, '003': 2, '004': 3, '005': 4, '006': 2, '007': 3, '008': 4,
                   '009': 5, '010': 6, '011': 7, '012': 8, '013': 9,
                   'TM': 2, 'T1A': 2, 'TB2': 2, 'TB3': 3, 'TB4': 4, 'T2C': 2, 'T3C': 3, 'T4C': 4,
                   'T5': 5, 'T6': 6, 'T7': 7, 'T8': 8, 'T9': 9}


def TypeDetect(dir_path):
    # 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if FLBZFlag == 1:
        yolo_type_dict = yolo_type_dict_gb
        dirname_type_dict = dirname_type_dict_gb
    else:
        yolo_type_dict = yolo_type_dict_mxg
        dirname_type_dict = dirname_type_dict_mxg

    # 2 选择要进行测试的路径
    if len(dir_path) == 0:
        dir_path = filedialog.askdirectory()
        if dir_path is None:
            return
    if PicDataGeShi == 0: # 跑标定数据时，需要手动选择标签路径
        lab_path = filedialog.askdirectory()
        if lab_path is None:
            return

    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if PicDataGeShi == 0:
        save_path2 = 'D:/目标标定iou_{:}_{:}'.format(PicDataType, curtime)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)

    if PicDataGeShi == 1:
        save_path2 = 'D:/识别错图pt_{:}_{:}'.format(PicDataType, curtime)
        if not os.path.exists(save_path2):
            os.mkdir(save_path2)

    if PicDataGeShi == 2:
        save_path2 = 'D:/原图分类pt_{:}'.format(PicDataType, curtime)
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
    CorrectNum_dict = {}
    ErrorNum_dict = {}
    totalNum = len(files)
    curNum = 0
    curCount = 0
    iouErrNum = 0
    hsjErrNum = [0,0,0,0,0]
    wxpNum = [0,0,0,0]
    llcNum = [0,0,0,0]
    t0 = time.time()

    imgs = []
    fileNames = []
    filePaths = []
    curPicNum = 0
    boxes_infos = []
    realboxes_lens = []
    realLabels = []
    realLabel_WXPs = []
    realLabel_LLCs = []

    for filePath in reversed(files):

        # 取得文件名（不带路径）
        fileName = Path(filePath).name
        # print(fileName) # 打印文件名

        # 5.1 从标签中读取类别和坐标
        if PicDataGeShi == 0:
            label_path = '{:}/{:}txt'.format(lab_path, fileName[0:len(fileName)-3])
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # 从lables文件中获取特定标签值的信息
                boxes_filtered1 = boxes[boxes[:, 0] == 21]  # hsj
                boxes_filtered2 = boxes[boxes[:, 0] == 14]  # 轮
                boxes_filtered = torch.cat((boxes_filtered1, boxes_filtered2), dim=0)
                # 去除labels列, 只保留坐标信息
                boxes_info = boxes_filtered[:, 1:]
                # 一共从标签文件中获取了多少个目标
                realboxes_len = boxes_info.size()[0]

        # 5.2 从文件夹上解析真实的车型
        leftRightFlag = 0;
        if PicDataGeShi == 1:
            realLabel = filePath[len(dir_path)+1 : len(dir_path)+4]  # 二级目录文件夹值：3位数字
            if not realLabel.isdigit():  # 若无法转换为数字, 则使用二级目录的前6位字符
                realLabel = filePath[len(dir_path)+1 : len(dir_path)+7]  # 二级目录文件夹值：6位标签
            if realLabel.find('连车右') != -1:  # 判断是否为 连车 文件夹
                realLabel = filePath[len(dir_path)+5 : len(dir_path)+8]  # 三级目录文件夹值：3位数字
                leftRightFlag = 2
            if realLabel.find('连车左') != -1:  # 判断是否为 连车 文件夹
                realLabel = filePath[len(dir_path)+5 : len(dir_path)+8]  # 三级目录文件夹值：3位数字
                leftRightFlag = 1

            if PicDataType == 3 or PicDataType == 7:  # 三位数字是 真危险品 文件夹，否则为 假危险品 文件夹
                if not realLabel.isdigit():
                    realLabel_WXP = 0
                else:
                    realLabel_WXP = 1

            if PicDataType == 4 or PicDataType == 8:  # 三位数字是 真冷链车 文件夹，否则为 假冷链车 文件夹
                if not realLabel.isdigit():
                    realLabel_LLC = 0
                else:
                    realLabel_LLC = 1

        # 5.3 原始数据, 无需解析
        if PicDataGeShi == 2:
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
            if PicDataGeShi == 0:
                boxes_infos = []
                realboxes_lens = []
            if PicDataGeShi == 1:
                realLabels = []
                if PicDataType == 3 or PicDataType == 7:
                    realLabel_WXPs = []
                if PicDataType == 4 or PicDataType == 8:
                    realLabel_LLCs = []

        imgs.append(im0)
        fileNames.append(fileName)
        filePaths.append(filePath)
        if PicDataGeShi == 0:
            boxes_infos.append(boxes_info)
            realboxes_lens.append(realboxes_len)
        if PicDataGeShi == 1:
            realLabels.append(realLabel)
            if PicDataType == 3 or PicDataType == 7:
                realLabel_WXPs.append(realLabel_WXP)
            if PicDataType == 4 or PicDataType == 8:
                realLabel_LLCs.append(realLabel_LLC)

        curPicNum += 1
        if curPicNum < myBatchSize:
            continue

        # 7 进行推理预测
        # 不要传入图片路径，否则图片数量越多，占用的内存越多，2万张图能吃光64G内存
        results = model.predict(source=imgs, conf=myConf, iou=myIou)

        # 8~11识别结果分析; 12识别率分析; 13未定义; 14按照要求存图
        for i in range(myBatchSize): #对batch中的每一张图进行分析
            det = results[i].boxes

            # 加速跑时, 若发现一个未识别的, 则再单独对这个样本做一次识别
            det_boxes = det.xywhn.squeeze()
            clsNum = det_boxes.size()[0]
            if myBatchSize > 1 and 0 == clsNum:
                rst = model.predict(source=imgs[i], conf=myConf, iou=myIou)
                det = rst[0].boxes

            fileName = fileNames[i]
            filePath = filePaths[i]

            if PicDataGeShi == 0:
                boxes_info = boxes_infos[i]
                realboxes_len = realboxes_lens[i]
            if PicDataGeShi == 1:
                realLabel = realLabels[i]
                if PicDataType == 3 or PicDataType == 7:
                    realLabel_WXP = realLabel_WXPs[i]
                if PicDataType == 4 or PicDataType == 8:
                    realLabel_LLC = realLabel_LLCs[i]

            recLabel = '0'
            maxConf = 0.0
            zcRecLabel = '0'
            zcMaxConf = 0.0
            wxpFlag = 0
            llcFlag = 0
            typeNum = 0
            mutiNames = {}
            vehNum = 0
            mutiVehs = {}

            # 8 识别结果分析（提取最优车型，是否是危险品, 是否冷链车)
            for d in det:
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                name = model.names[int(cls)]

                # 8.0 排除肯定不是的目标区域
                paichuFlag = 0
                if name != 'wxp' and name != 'llc' and name != 'lun' and name != 'hsj':
                    typebox = d.xyxyn.squeeze()
                    typebox = typebox.cpu().numpy()
                    # 若目标区域在左右边界，且特别的小, 则不信任结果
                    if (typebox[0] > 0.75 or typebox[2] < 0.25) \
                            and typebox[2] - typebox[0] < typebox[3] - typebox[1]:
                        if conf > 0.75 and (typebox[2] - typebox[0])*imgs[i].width > 200:
                            paichuFlag = 0
                        else:
                            paichuFlag = 1
                            print(f'左右边界：{fileName}')
                    # 若目标区域特别靠上，则不信任结果
                    elif typebox[3] < 0.55 and typebox[1] < 0.55:
                        paichuFlag = 2
                        print(f'隔壁车道：{fileName}')
                    # 若目标区域特别靠下，则不信任结果
                    elif typebox[3] > 0.82 and typebox[1] > 0.82:
                        paichuFlag = 3
                        print(f'底部遮挡：{fileName}')
                else:
                    paichuFlag = 4

                # 8.1 提取最优车型
                if paichuFlag == 0:
                    mutiNames[typeNum] = name
                    typeNum = typeNum + 1

                    if vehNum != 0:
                        for k in range(vehNum):
                            boxk = mutiVehs[k].xywhn.cpu().numpy()
                            boxd = d.xywhn.cpu().numpy()
                            if bbox_iou_xywh(boxk.squeeze(), boxd.squeeze()) < 0.85:
                                mutiVehs[vehNum] = d
                                vehNum = vehNum + 1
                    else:
                        mutiVehs[vehNum] = d
                        vehNum = vehNum + 1

                    if conf > maxConf:
                        recLabel = name  # 最优车型
                        maxConf = conf
                        recbox = typebox

                # 8.2 是否专项作业车
                if paichuFlag == 0:
                    if int(cls) >= 21 and int(cls) <= 26:
                        zcRecLabel = model.names[int(cls)]
                        zcMaxConf = conf
                        zcbox = typebox

                # 8.3 是否危险品车辆
                if name == 'wxp':
                    wxpFlag = 1
                    wxpbox = d.xyxyn.squeeze()
                    wxpbox = wxpbox.cpu().numpy()

                # 8.4 是否冷链车车辆
                if name == 'llc':
                    llcFlag = 1
                    llcbox = d.xyxyn.squeeze()
                    llcbox = llcbox.cpu().numpy()

            # 8.5 去除干扰车后, 若仍有两个车型, 则按下面的方法处理
            yichuliFlag = 0
            if vehNum >= 2:
                # 8.5.1 若是专车拖着一个车, 则修改车型值为专车结果
                if leftRightFlag==0 and zcRecLabel != '0' and zcRecLabel != recLabel and recbox[0] > zcbox[2]:
                    recLabel = zcRecLabel  # 最优车型
                    maxConf = zcMaxConf
                    recbox = zcbox
                    yichuliFlag = 1

                # 8.5.2 若两个车型的IOU重合度很高, 则取大权重的那个车型
                # 不用编码(前面的逻辑已实现)

                # 8.5.3 若是普通车拖着一个挂车, 则修改车型值为含挂车的那个结果
                #       方法: 一侧重合度高, 另一侧不重合的, 且面积小的那个是(客1, 货1, 货2), 则取面积大的那个车型
                if vehNum == 2:
                    box0 = mutiVehs[0].xyxyn.squeeze()
                    box0 = box0.cpu().numpy()
                    box1 = mutiVehs[1].xyxyn.squeeze()
                    box1 = box1.cpu().numpy()
                    name0 = model.names[int(mutiVehs[0].cls.squeeze())]
                    typeXGB0 = yolo_type_dict_gb.get(name0)
                    name1 = model.names[int(mutiVehs[1].cls.squeeze())]
                    typeXGB1 = yolo_type_dict_gb.get(name1)
                    if (abs(box0[0] - box1[0]) < 0.05 and abs(box0[2] - box1[2]) > 0.20) \
                            or (abs(box0[2] - box1[2]) < 0.05 and abs(box0[0] - box1[0]) > 0.20):
                        if (box0[0] - box1[0] > 0.2 or box1[2] - box0[2] > 0.2) and (typeXGB0==1 or typeXGB0==11 or typeXGB0==12):
                            recLabel = name1
                            maxConf = mutiVehs[1].conf.squeeze()
                            recbox = box1
                            yichuliFlag = 1
                        if (box1[0] - box0[0] > 0.2 or box0[2] - box1[2] > 0.2) and (typeXGB1==1 or typeXGB1==11 or typeXGB1==12):
                            recLabel = name0
                            maxConf = mutiVehs[0].conf.squeeze()
                            recbox = box0
                            yichuliFlag = 1

                # 8.5.4 若几乎完全不重合, 则只返回第一个结果
                #       方法: 寻找第一个车型, 若它跟最优车型几乎完全不重合, 则修改最优车型
                if yichuliFlag == 0:
                    for k in range(vehNum):
                        boxk = mutiVehs[k].xyxyn.squeeze()
                        boxk = boxk.cpu().numpy()
                        iou = cal_iou(boxk.squeeze(), recbox.squeeze())
                        if leftRightFlag == 1:  # 连车左
                            if boxk[0] < recbox[0] and abs(boxk[3] - recbox[3]) < 0.15 and iou < 0.15:
                                recLabel = model.names[int(mutiVehs[k].cls.squeeze())]
                                maxConf = mutiVehs[k].conf.squeeze()
                                recbox = boxk
                        if leftRightFlag == 2:  # 连车右
                            if boxk[2] > recbox[2] and abs(boxk[3] - recbox[3]) < 0.15 and iou < 0.15:
                                recLabel = model.names[int(mutiVehs[k].cls.squeeze())]
                                maxConf = mutiVehs[k].conf.squeeze()
                                recbox = boxk

            # 8.6 去除隔壁车道的车误判为 危险品 的情况
            typeXGB = yolo_type_dict.get(recLabel)
            if wxpFlag:
                if recLabel != '0':
                    iou = cal_iou(typebox, wxpbox)
                    if iou < 0.25 or typeXGB < 11 or typeXGB > 16:
                        wxpFlag = 0
                else:
                    wxpFlag = 0 # 若只是别为wxp，但未识别到车型，则也不信任wxp

            # 8.7 去除隔壁车道的车误判为 冷链车 的情况
            if llcFlag:
                if recLabel != '0':
                    iou = cal_iou(typebox, llcbox)
                    if iou < 0.25 or typeXGB < 11 or typeXGB > 16:
                        llcFlag = 0
                else:
                    llcFlag = 0 # 若只是别为llc，但未识别到车型，则也不信任llc

            # 9 识别结果分析（识别了几个轮, 是否检到HSJ了）
            hsjNum = 0
            hsjconf = 0.0
            if recTypeFlag == 3 or recTypeFlag == 4:
                for d in det:
                    cls, conf = d.cls.squeeze(), d.conf.squeeze()
                    name = model.names[int(cls)]
                    if name == 'lun':  # 识别为车轮
                        lunNum = lunNum + 1
                    if name == 'hsj':  # 识别为车轮
                        hsjNum = hsjNum + 1
                        hsjconf = conf

            # 10 识别结果分析（轮的IOU分析）
            iouErrFlag = 0
            if recTypeFlag == 3:
                det_boxes = det.xywhn.squeeze()
                recboxes_len = det_boxes.size()[0]
                # gpu转cpu 不转的话numpy会报错
                boxes_info = boxes_info.cpu().numpy()
                det_boxes = det_boxes.cpu().numpy()
                # iou原始值的写法  iou = bbox_iou_xywhn(det_boxes, boxes_info)
                if realboxes_len != 0 and recboxes_len != 0:
                    iou = bbox_iou_xywhn(det_boxes, boxes_info, set_threshold=True, threshold=0.5)
                    total_sum = np.sum(iou)
                    if realboxes_len != recboxes_len or realboxes_len != total_sum:
                        iouErrFlag = 1
                        iouErrNum = iouErrNum + 1
                else:
                    iouErrFlag = 1
                    iouErrNum = iouErrNum + 1

            # 11 识别结果分析（是否识别对了）
            if PicDataGeShi == 1:
                save_flage = 0
                if recTypeFlag == 2:  # 仅车型
                    recLabel_xgb = yolo_type_dict.get(recLabel) if recLabel != '0' else 0
                    realLabel_xgb = dirname_type_dict.get(realLabel)

                    if recLabel_xgb is None or realLabel_xgb is None:
                        save_flage = 1
                    elif recLabel_xgb == 0 and realLabel_xgb == 1:
                        # 说明: 识别失败时, 若发现图的宽度很窄, 则强行将其默认为客1, 可以提高识别率
                        if imgs[i].width > 1000:
                            save_flage = 1
                    elif recLabel_xgb != realLabel_xgb:
                        # 说明: 该逻辑对 蓝牌 和 黄牌 是有效的, 但 新能源牌 和 白牌车辆 可能会出错
                        if not ((recLabel_xgb == 12 and realLabel_xgb == 11)
                                or (recLabel_xgb == 11 and realLabel_xgb == 12)
                                or (recLabel_xgb == 1 and realLabel_xgb == 2)
                                or (recLabel_xgb == 2 and realLabel_xgb == 1)):
                            save_flage = 1

                if recTypeFlag == 3:  # 仅车轮
                    lunCount = dirname_lun_dict.get(realLabel)
                    recLabel = '{:}'.format(lunNum)  # 给识别标签赋值
                    if lunNum != lunCount:
                        save_flage = 1

                if recTypeFlag == 4:  # HSJ轮
                    save_flage = 1 # TODO


            # 12 各个类别的识别率统计
            if PicDataGeShi == 1:
                if realLabel not in CorrectNum_dict:
                    CorrectNum_dict.update({realLabel: 0})
                    ErrorNum_dict.update({realLabel: 0})
                # 修改字典的值
                if save_flage:
                    ErrorNum_dict[realLabel] = ErrorNum_dict.get(realLabel) + 1
                else:
                    CorrectNum_dict[realLabel] = CorrectNum_dict.get(realLabel) + 1

            # 13 ###

            # 14 按指定的要求存图
            # 14.1 iou错图
            if PicDataGeShi == 0:
                save_ImgPath = '{:}/{:}'.format(save_path2, fileName)
                save_LabPath = '{:}/{:}txt'.format(save_path2, fileName[0:len(fileName) - 3])
                if iouErrFlag:
                    shutil.copyfile(filePath, save_ImgPath)
                    shutil.copyfile(label_path, save_LabPath)

            # 14.2 错误存图
            if PicDataGeShi == 1:
                nname = realLabel + '识别为' + recLabel
                save_path1 = '{:}/{:}'.format(save_path2, nname)
                if not os.path.exists(save_path1) and save_flage == 1:
                    os.mkdir(save_path1)
                save_path = '{:}/{:}/{:}'.format(save_path2, nname, fileName)
                if save_flage:
                    shutil.copyfile(filePath, save_path)

            # 14.3 后视镜错存图
            if PicDataGeShi == 3:
                if hsjNum == 1 and hsjconf > 0.5:
                    hsjErrNum[4] = hsjErrNum[4] + 1
                else:
                    hsjn = hsjNum if hsjNum <= 3 else 3
                    hsjErrNum[hsjn] = hsjErrNum[hsjn] + 1
                    save_path1 = 'D:/hsj{:}'.format(hsjNum)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = '{:}/{:}'.format(save_path1, fileName)
                    shutil.copyfile(filePath, save_path)

            # 14.4 多车型存图（当识别出多个目标，但标签又不一样时）
            if PicDataGeShi != 0 and typeNum > 1 and 0:
                laba = yolo_type_dict.get(mutiNames[0])
                labb = yolo_type_dict.get(mutiNames[1])
                if laba != labb:
                    save_path1 = 'D:/多个车型值_{:}_{:}'.format(PicDataType, curtime)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = 'D:/多个车型值_{:}_{:}/{:}'.format(PicDataType, curtime, fileName)
                    shutil.copyfile(filePath, save_path)     # 保存原始图像
                    model.predict(source=filePath, save=True)  # 保存画框图像（直接画没试出来，只能重新预测一遍了）

            # 14.5 危险品错误存图
            if PicDataType == 3 or PicDataType == 7:
                # 识别结果 和 真实结果 一样时，则正确数加1，否则将其保存出来, 用于模型训练
                if realLabel_WXP == wxpFlag:
                    if wxpFlag:
                        wxpNum[0] = wxpNum[0] + 1
                    else:
                        wxpNum[1] = wxpNum[1] + 1
                else:
                    if realLabel_WXP:
                        wxpNum[2] = wxpNum[2] + 1
                    else:
                        wxpNum[3] = wxpNum[3] + 1
                    save_path1 = '{:}/危险{:}'.format(save_path2, realLabel_WXP)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = '{:}/{:}'.format(save_path1, fileName)
                    shutil.copyfile(filePath, save_path)

            # 14.6 危险品样本累积
            if PicDataType == 1 or PicDataType == 5:
                wxpName = filePath[len(filePath) - 6:len(filePath)]
                if wxpFlag == 1:
                    wxpNum[0] = wxpNum[0] + 1
                zb = wxpNum[0] / (1 + curCount)
                # 若是危险品，且名称中不含wxp标记，且在目录中的占比很低，则将其保存出来, 用于样本收集
                if wxpFlag != 0 and wxpName != 'wx.jpg' and zb < 0.1:
                    save_path1 = 'D:/样本累积/wxp{:}'.format(PicDataType)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = 'D:/样本累积/wxp{:}/{:}'.format(PicDataType, fileName)
                    shutil.copyfile(filePath, save_path)
            wxpFlag = 0

            # 14.7 冷链车错误存图
            if PicDataType == 4 or PicDataType == 8:
                # 识别结果 和 真实结果 一样时，则正确数加1，否则将其保存出来, 用于模型训练
                if realLabel_LLC == llcFlag:
                    if llcFlag:
                        llcNum[0] = llcNum[0] + 1
                    else:
                        llcNum[1] = llcNum[1] + 1
                else:
                    if realLabel_LLC:
                        llcNum[2] = llcNum[2] + 1
                    else:
                        llcNum[3] = llcNum[3] + 1
                    save_path1 = '{:}/冷链{:}'.format(save_path2, realLabel_LLC)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = '{:}/{:}'.format(save_path1, fileName)
                    shutil.copyfile(filePath, save_path)

            # 14.8 冷链车样本累积
            if PicDataType == 1 or PicDataType == 5:
                llcName = filePath[len(filePath) - 6:len(filePath)]
                if llcFlag == 1:
                    llcNum[0] = llcNum[0] + 1
                zb = llcNum[0] / (1 + curCount)
                # 若是冷链车，且名称中不含llc标记，且在目录中的占比很低，则将其保存出来, 用于样本收集
                if llcFlag != 0 and llcName != 'll.jpg' and zb < 0.1:
                    save_path1 = 'D:/样本累积/llc{:}'.format(PicDataType)
                    if not os.path.exists(save_path1):
                        os.mkdir(save_path1)
                    save_path = 'D:/样本累积/llc{:}/{:}'.format(PicDataType, fileName)
                    shutil.copyfile(filePath, save_path)
            llcFlag = 0

            # 14.9 分类存图
            if PicDataGeShi == 2:
                save_path1 = '{:}/{:}'.format(save_path2, recLabel)
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)
                save_path = '{:}/{:}'.format(save_path1, fileName)
                #if maxConf > 0.93: # 需要权重过滤时
                # shutil.copyfile(filePath, save_path)
                shutil.move(filePath, save_path)

        # 15 每隔100张图片打印一次识别率, 既可以看进度，也可以判定是否还要继续运行
        curNum = curNum + myBatchSize
        curCount = curCount + myBatchSize
        if curNum >= 100:
            curNum = 0
            errNum = 0
            for key in CorrectNum_dict.keys():
                errNum += ErrorNum_dict.get(key)
                Correctpercent = [key, '错{:}'.format(ErrorNum_dict.get(key)),
                                  '对{:}'.format(CorrectNum_dict.get(key)),
                                  '总{:}'.format(CorrectNum_dict.get(key) + ErrorNum_dict.get(key)),
                                  '率{:.2%}'.format(CorrectNum_dict.get(key) / (
                                          CorrectNum_dict.get(key) + ErrorNum_dict.get(key)))]
                if PicDataType == 1 or PicDataType == 5:
                    print(Correctpercent)

            if PicDataType == 3 or PicDataType == 7:
                print(f'危险品：{curCount}/{totalNum}，对({wxpNum[0]}，{wxpNum[1]})，错({wxpNum[2]}，{wxpNum[3]})，'
                      f'率({wxpNum[0] / (1 + wxpNum[0] + wxpNum[2]):.2%}，{wxpNum[1] / (1 + wxpNum[1] + wxpNum[3]):.2%})，'
                      f'{(wxpNum[0] + wxpNum[1]) / curCount:.2%}')

            if PicDataType == 4 or PicDataType == 8:
                print(f'冷链车：{curCount}/{totalNum}，对({llcNum[0]}，{llcNum[1]})，错({llcNum[2]}，{llcNum[3]})，'
                      f'率({llcNum[0] / (1 + llcNum[0] + llcNum[2]):.2%}，{llcNum[1] / (1 + llcNum[1] + llcNum[3]):.2%})，'
                      f'{(llcNum[0] + llcNum[1]) / curCount:.2%}')

            t = int((time.time() - t0) * (totalNum - curCount) / curCount)
            h = t // 3600
            m = (t % 3600) // 60
            s = t % 60
            t1 = int(time.time() - t0)
            h1 = t1 // 3600
            m1 = (t1 % 3600) // 60
            s1 = t1 % 60

            if errNum == 0:
                errNum = hsjErrNum[0] + hsjErrNum[2] + hsjErrNum[3]
                print(f'{curCount}/{totalNum}，错{errNum}={hsjErrNum[0]}+{hsjErrNum[2]}+{hsjErrNum[3]}，'
                      f'低{hsjErrNum[1]}，高{hsjErrNum[4]}，总率{(curCount - errNum) / curCount:.2%}   '
                      f'已耗时间：{h1}时{m1}分{s1}秒，剩余时间：{h}时{m}分{s}秒\n')
            else:
                print(f'{curCount}/{totalNum}，总错{errNum}，总率{(curCount - errNum) / curCount:.2%}   '
                      f'已耗时间：{h1}时{m1}分{s1}秒，剩余时间：{h}时{m}分{s}秒\n')

    # 16 识别率计算、显示并写入文件中
    if len(save_path2) > 0:
        errNum = 0
        save_path = '{:}/{:}'.format(save_path2, '识别率.txt')
        file = open(save_path, 'w')
        for key in CorrectNum_dict.keys():
            errNum += ErrorNum_dict.get(key)
            tmpStr = '{:}, 错{:}, 对{:}, 总{:}, 率{:.2%}\n'.format(key, ErrorNum_dict.get(key),
                 CorrectNum_dict.get(key), CorrectNum_dict.get(key) + ErrorNum_dict.get(key),
                 CorrectNum_dict.get(key) / (CorrectNum_dict.get(key) + ErrorNum_dict.get(key)))
            print(tmpStr)
            file.write(tmpStr)

        if PicDataType == 3 or PicDataType == 7:
            tmpStr = '危险品：{:}/{:}，对({:}，{:})，错({:}:{:})，率({:.2%}，{:.2%})，{:.2%}\n'.format(
                curCount, totalNum, wxpNum[0], wxpNum[1], wxpNum[2], wxpNum[3], wxpNum[0] / (1 + wxpNum[0] + wxpNum[2]),
                wxpNum[1] / (1 + wxpNum[1] + wxpNum[3]), (wxpNum[0] + wxpNum[1]) / curCount)
            print(tmpStr)
            file.write(tmpStr)

        if PicDataType == 4 or PicDataType == 8:
            tmpStr = '冷链车：{:}/{:}，对({:}，{:})，错({:}:{:})，率({:.2%}，{:.2%})，{:.2%}\n'.format(
                curCount, totalNum, llcNum[0], llcNum[1], llcNum[2], llcNum[3], llcNum[0] / (1 + llcNum[0] + llcNum[2]),
                llcNum[1] / (1 + llcNum[1] + llcNum[3]), (llcNum[0] + llcNum[1]) / curCount)
            print(tmpStr)
            file.write(tmpStr)

        tmpStr = '总错{:}，总数{:}，总率{:.2%}，总耗时：{:.2f}s\n'.format(
            errNum, totalNum, (totalNum - errNum) / totalNum, time.time() - t0)
        print(tmpStr)
        file.write(tmpStr)
        file.close()




if __name__ == "__main__":
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    saveName = "train{:}".format(curtime)

    # 1 加载模型
    if modelSelect == 1:
        model_path = filedialog.askopenfilename()
        if model_path is None:
            print("请选择一个模型文件(pt)，否则无法进行推理！")
        else:
            model = YOLO(model_path)
    if recTypeFlag == 2:
        if modelSelect == 2:
            model = YOLO("E:/DLModel/curWeights/best-v8.pt")
        if modelSelect == 3:
            model = YOLO("E:/DLModel/curWeights/best-v10.pt")
        if modelSelect == 4:
            model = YOLO("E:/DLModel/curWeights/best-v11.pt")
    if recTypeFlag == 3:
        if modelSelect == 2:
            model = YOLO("E:/DLModel/curWeights/best-g3-lunFZ-320-v8.pt")
        if modelSelect == 3:
            model = YOLO("E:/DLModel/curWeights/best-g3-lunFZ-320-v10.pt")
    if recTypeFlag == 4:
        if modelSelect == 2:
            model = YOLO("E:/DLModel/curWeights/best-g3-HsjLun-320-v8.pt")
        if modelSelect == 3:
            model = YOLO("E:/DLModel/curWeights/best-g3-HsjLun-320-v10.pt")

    # 2 推理1：通过对话框选择样本路径，逐个识别并统计识别率
    if runControl == 4:
        TypeDetect("")

    # 3 推理2：指定多个待识别目录路径，逐个识别并统计识别率
    if runControl == 5:
        if PicDataType <= 4:
            my_list = ['D:/0车型样本/高清 易错样本-6千',
                        'D:/0车型样本/高清 特殊车辆-1千',
                        'D:/0车型样本/高清 非车干扰-6千',
                        'D:/0车型样本/高清 训练样本-1万',
                        'D:/0车型样本/高清 拼图不好-2万',
                        'D:/0车型样本/高清 墨西哥型-3万',
                        'D:/0车型样本/高清.危险品车-9千',
                        'D:/0车型样本/高清.冷链车辆-1万',
                        'D:/0车型样本/高清 客货专车（26万）',
                        'D:/0车型样本/高清 客一货六（37万）']
            # my_list = ['D:/0车型样本/标清 拼图不好-8千',
            #            'D:/0车型样本/标清 普通车辆-客一货六（16万）',
            #            'D:/0车型样本/标清 普通车辆-所有车型（16万）',
            #            'D:/0车型样本/标清 特殊车辆-1千',
            #            'D:/0车型样本/标清 训练样本-1万']
            for ele in my_list:
                if ele.find('危险品') != -1:
                    PicDataType = 3
                elif ele.find('冷链车') != -1:
                    PicDataType = 4
                else:
                    PicDataType = 1
                if ele.find('墨西哥') != -1:
                    FLBZFlag = 2
                else:
                    FLBZFlag = 1
                TypeDetect(ele)
        elif PicDataType > 4 and PicDataType <= 8:
            my_list = ['H:/0-门架样本(全部车型)',
                       'H:/0-门架样本(客一货六)']
            for ele in my_list:
                TypeDetect(ele)
        else:
            print("请选择正确的样本类型！")


