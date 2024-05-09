import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile

# 控制变量
runControl  = 4     # 运行模式：1-存原始图, 2-存画框图(检测框), 3-存画框图(标签框), 4-存画框图(差异框)，5-追加标定

myConf = 0.35
myIou = 0.55

model = None        # 单个模型

def yolo_txt2cv_points(file_path,save_path):#前者是需要的文件,文件包含两个文件夹images和labels 后者是需要保存在哪
    images_path = os.path.join(file_path,'images')
    labels_path = os.path.join(file_path,'labels')
    print(len(os.listdir(images_path)))

    for file in os.listdir(images_path):
        image_path = os.path.join(images_path,file)
        image = cv2.imread(image_path)
        file_name = file.split('.')[0]
        label_file = file_name + '.txt'
        label_path = os.path.join(labels_path, label_file)
        width = image.shape[1]
        height = image.shape[0]
        with open(label_path, 'r') as f:
            rects = f.readlines()
            for rect in rects:
                a = float(rect[2:10])
                b = float(rect[11:19])
                c = float(rect[20:28])
                d = float(rect[29:37])
                box_width = c*width
                box_height = d*height
                first_point = (int(a*width - box_width/2), int(b*height - box_height/2))
                last_point = (int(a*width + box_width/2), int(b*height + box_height/2))
                # print(first_point,last_point)
                cv2.rectangle(image, first_point, last_point, (0, 255, 0), 2)
                save = os.path.join(save_path,file)
                # print(save)
                cv2.imwrite(save,image)
                # print(a,b,c,d)




def cal_iou(box1, box2):
    """
    :param box1: = [left1, top1, right1, bottom1]
    :param box2: = [left2, top2, right2, bottom2]
    :return:
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    # 计算每个矩形的面积
    s1 = (right1 - left1) * (bottom1 - top1)  # b1的面积
    s2 = (right2 - left2) * (bottom2 - top2)  # b2的面积

    # 计算相交矩形
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)

    # 相交框的w,h
    w = max(0, right - left)
    h = max(0, bottom - top)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou

# xywhn 格式计算
# xywhn 标记格式：中心坐标，宽度和高度
# xyxyn 标记格式：左上坐标，右下坐标
# 后缀n 表示已经归一化处理
def bbox_iou_xywhn(box1, box2, set_threshold=None, threshold=0.75):
    #如果box1,box2是一维数组,就复制成2维数组
    if box1.ndim == 1:
        box1 = np.tile(box1[np.newaxis, :], (2, 1))
    if box2.ndim == 1:
        box2 = np.tile(box2[np.newaxis, :], (2, 1))

    # 将box1和box2转换回xyxy格式
    box1_xy = box1[..., :2] - (box1[..., 2:] / 2.)
    box1_wh = box1[..., 2:]
    box1_x1y1 = np.concatenate((box1_xy, box1_xy + box1_wh), axis=-1)
    box2_xy = box2[..., :2] - (box2[..., 2:] / 2.)
    box2_wh = box2[..., 2:]
    box2_x1y1 = np.concatenate((box2_xy, box2_xy + box2_wh), axis=-1)

    # 计算交集的左上角和右下角点
    xy1 = np.maximum(box1_x1y1[:, None, :2], box2_x1y1[..., :2])
    xy2 = np.minimum(box1_x1y1[:, None, 2:], box2_x1y1[..., 2:])
    # 计算交集区域的宽和高
    wh = np.maximum(xy2 - xy1, 0)
    # 计算交集区域的面积
    inter_area = wh[..., 0] * wh[..., 1]
    # 计算并集区域的面积
    box1_area = box1_wh.prod(axis=-1)
    box2_area = box2_wh.prod(axis=-1)
    union_area = box1_area[:, None] + box2_area - inter_area
    # 计算IoU并返回结果
    iou = inter_area / union_area

    # 大于阈值的置成1，小于阈值的置成0
    if set_threshold:
        iou[iou >= threshold] = 1
        iou[iou < threshold] = 0

    return iou

def TypeDetect():
    # 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 1 选择要进行测试的路径
    dir_path = filedialog.askdirectory()
    if dir_path is None:
        return
    lab_path = filedialog.askdirectory()
    if lab_path is None:
        return

    # 2 创建存储错图的文件夹
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path2 = 'D:/目标标定iou_{:}'.format(curtime)
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
            if os.path.getsize(label_path) > 0:
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # 去除class_labels的标记列
                boxes_info = boxes[:, 1:]
                # 标签文件中class_labels的数量
                realboxes_len = boxes_info.size()[0]
            else:
                realboxes_len = 0
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

        # 7 进行推理预测
        # 不要传入图片路径，否则图片数量越多，占用的内存越多，2万张图能吃光64G内存
        results = model.predict(source=im0, conf=myConf, iou=myIou)

        # 8 识别结果分析(找出:多检的,漏检的和错判的,存图)
        iouErrFlag = 0
        for rst in reversed(results):  # TODO 这里允许一次处理多张图片
            det = rst.boxes

            det_boxes = det.xywhn
            recboxes_len = det_boxes.size()[0]

            # gpu转cpu 不转的话numpy会报错
            if (realboxes_len != 0):
                boxes_info = boxes_info.cpu().numpy()
            if (recboxes_len != 0):
                det_boxes = det_boxes.cpu().numpy()

            # iou原始值的写法  iou = bbox_iou_xywhn(det_boxes, boxes_info)
            if realboxes_len != 0 and recboxes_len != 0:
                iou = bbox_iou_xywhn(det_boxes, boxes_info, set_threshold=True, threshold=0.5)
                total_sum = np.sum(iou)
                if realboxes_len != recboxes_len or realboxes_len != total_sum:
                    iouErrFlag = 1
                    iouErrNum = iouErrNum + 1
            else:
                if realboxes_len == 0 and recboxes_len == 0:
                    iouErrFlag = 2
                else:
                    iouErrFlag = 1
                    iouErrNum = iouErrNum + 1

        # 9 按指定的要求存图
        save_ImgPath = '{:}/{:}'.format(save_path2, fileName)
        save_LabPath = '{:}/{:}txt'.format(save_path2, fileName[0:len(fileName) - 3])

        # 9.1 提取标定和检测不一样的原始数据
        if iouErrFlag == 1 and runControl == 1:
            shutil.copyfile(filePath, save_ImgPath)
            if (realboxes_len != 0):
                shutil.copyfile(label_path, save_LabPath)

        # 9.2 提取标定和检测不一样的画框图(检测框)
        if iouErrFlag == 1 and runControl == 2:
            for r in results:
                im_array = r.plot()  # 绘制预测结果的BGR numpy数组
                img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)
            # cv2_imshow(img)  # 显示图像
            cv2.imwrite(save_ImgPath, img)

        # 9.2 提取标定和检测不一样的画框图(标签框)
        if iouErrFlag == 1 and runControl == 3:
            width = im0.width
            height = im0.height
            if (realboxes_len != 0):
                for r in boxes_info:
                    first_point = (int(r[0]*width - r[2]*width/2), int(r[1]*height - r[3]*height/2))
                    last_point = (int(r[0]*width + r[2]*width/2), int(r[1]*height + r[3]*height/2))
                    cv2.rectangle(im0, first_point, last_point, (0, 255, 0), 2)
            cv2.imwrite(save_ImgPath,im0)

        # 9.3 提取标定和检测不一样的画框图(差异框)
        if iouErrFlag == 1 and runControl == 4: #
            for r in results:
                im_array = r.plot()  # 绘制预测结果的BGR numpy数组
                img = cv2.cvtColor(im_array[..., ::-1], cv2.COLOR_BGR2RGB)

            width = im0.width
            height = im0.height
            if (realboxes_len != 0):
                for r in boxes_info:
                    center = (int(r[0]*width), int(r[1]*height))
                    radius = int((int(r[2]*width) + int(r[3]*height)) / 4)
                    cv2.circle(img, center, radius, (0, 255, 0), 2)

            # cv2_imshow(img)  # 显示图像
            cv2.imwrite(save_ImgPath, img)

        # 9.4 给标定文件追加新检测出来的目标
        if runControl == 5:
            with open(label_path, 'r') as f:
                data = f.readlines()
            with open(save_LabPath, 'w') as f:
                # 逐行写入原内容
                for line in data:
                    f.write(line)

                # 追加新目标写入
                boxes = results[0].boxes
                for box in boxes:
                    cat_num = int(box.cls.cpu())
                    if cat_num == 1 or cat_num == 3:
                        label = box.xywhn.cpu().numpy()
                        size = label[0].tolist()
                        size_string = ' '.join(map(str, size))
                        result = f'{cat_num} {size_string}\n'
                        f.write(str(result))

        # 10 打印进度
        print(f'进度：{iouCurNum}/{totalNum}, 错{iouErrNum}, '
              f'率{iouCurNum / (iouErrNum+iouCurNum):.2%}  ' + fileName)



if __name__ == "__main__":
    # 1 加载模型
    model_path = filedialog.askopenfilename()
    if model_path is None:
        print("请选择一个模型文件(pt)，否则无法进行推理！")
    else:
        model = YOLO(model_path)

    # model = YOLO("E:\\DLRuns\\detectV8\\train20240415_083813\\weights\\best.pt")

    # 2 通过对话框选择样本路径，逐个识别，存错误图
    TypeDetect()




