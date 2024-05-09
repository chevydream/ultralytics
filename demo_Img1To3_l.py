import os, time
# import cutimage
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFile

def Img1To3():
    #lrtb->xywh
    def lrtb2xywh(left,right,top,bottom,width,height):
        x = (left+right)/(2*width)
        y = (top+bottom)/(2*height)
        w = (right-left)/width
        h = (bottom-top)/height
        return x,y,w,h

    # 解决错误：IOError: broken data stream when reading image file
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 1 选择要进行分割的图片及标签路径
    dir_path = filedialog.askdirectory()
    if dir_path is None:
        return
    lab_path = filedialog.askdirectory()
    if lab_path is None:
        return
    #dir_path="K:/trainimages/test/images"
    #lab_path="K:/trainimages/test/labels"

    # 2 创建存储分割图片和标签的文件夹
    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    Imgsavepath = dir_path + curtime
    os.makedirs(Imgsavepath, exist_ok=True)
    labsavepath = lab_path + curtime
    os.makedirs(labsavepath, exist_ok=True)

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
    for filePath in reversed(files):
        # 4 取得文件名（不带路径）
        fileName = Path(filePath).name
        # 5 从标签中读取类别和坐标
        label_path = '{:}/{:}txt'.format(lab_path, fileName[0:len(fileName)-3])
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
            # 标签文件中class_labels的数量
                realboxes_len = len(lines)
        else:
            realboxes_len = 0
        if realboxes_len>0:
            # 6 读取图片到im0中
            readtype = 0
            if readtype == 0:
                with open(filePath, 'rb') as f:
                    f = f.read()    # 这样读就是二进制的
                if len(f) < 1024:
                    print(filePath) # 打印一下异常图的文件名
                    continue
                f = f + B'\xff' + B'\xd9'   # 这句是补全数据的
                im0 = Image.open(BytesIO(f))
            else:
                im0 = Image.open(filePath)# 不安全，遇到没ffd9结尾的图，程序后面会出错，故改成了后面的样式！！！
            if im0.mode != "RGB":
                im0 = im0.convert('RGB')

            # 7 分割成3张图
            #计算分割尺寸 8的倍数
            nwidth =round(im0.width/3)-round(im0.width/3)%8
            # 计算裁剪后的区域  (left, upper, right, bottom)
            crop_region1 = (0, 0, nwidth, im0.height)
            crop_region2 = (nwidth, 0, nwidth*2, im0.height)
            crop_region3 = (nwidth*2, 0, im0.width, im0.height)
            # 裁剪图像
            CutImg1 = im0.crop(crop_region1)
            CutImg2 = im0.crop(crop_region2)
            CutImg3 = im0.crop(crop_region3)

            # 保存图像
            CutImg1.save(os.path.join(Imgsavepath,fileName.split(".")[0] + 'CutImg1' + '.jpg'))
            CutImg2.save(os.path.join(Imgsavepath,fileName.split(".")[0] + 'CutImg2' + '.jpg'))
            CutImg3.save(os.path.join(Imgsavepath,fileName.split(".")[0] + 'CutImg3' + '.jpg'))
            coords_flags = [[], [], []]  # 分别存储三张分割图片的标签数据

            # 8 标签分成3份
            for line in lines:
                line = line.strip().split()
                class_index = int(line[0])
                x, y, w, h = map(float, line[1:])
                #坐标转换 xywh->ltrb
                # 计算标签框的坐标
                left = int((x - w / 2) * im0.width)
                top = int((y - h / 2) * im0.height)
                right = int((x + w / 2) * im0.width)
                bottom = int((y + h / 2) * im0.height)
                #标签筛选参数:标签宽度
                width=15
                if right<nwidth:#标签在最左边的图
                    #coordsflage = [class_index, left, right, top, bottom] # 实际坐标
                    x, y, w, h = lrtb2xywh(left, right, top, bottom, CutImg1.width, CutImg1.height)
                    coords_flags[0].append([class_index, (x, y, w, h)])
                if left<=nwidth and nwidth<=right:#标签在图1 2上都有
                    if abs(left-nwidth)>width: #sl 表示边界左侧的图 此处为图1
                        x, y, w, h = lrtb2xywh(left, nwidth, top, bottom, CutImg1.width, CutImg1.height)
                        coords_flags[0].append([class_index, (x, y, w, h)])
                    if abs(right-nwidth)>width: #sr 表示边界右侧的图 此处为图2
                        x, y, w, h = lrtb2xywh(1, right-nwidth, top, bottom, CutImg2.width, CutImg2.height)
                        coords_flags[1].append([class_index, (x, y, w, h)])
                if nwidth<left and right<nwidth*2: #标签在图2上
                    x, y, w, h = lrtb2xywh(left-nwidth, right - nwidth, top, bottom, CutImg2.width, CutImg2.height)
                    coords_flags[1].append([class_index, (x, y, w, h)])
                if left<nwidth*2 and nwidth*2<right:#标签在图2 3上都有
                    if abs(left-nwidth) > width:  # sl 表示边界左侧的图 此处为图2
                        x, y, w, h = lrtb2xywh(left - nwidth, nwidth, top, bottom, CutImg2.width, CutImg2.height)
                        coords_flags[1].append([class_index, (x, y, w, h)])
                    if abs(right-nwidth) > width:  # sr 表示边界右侧的图 此处为图3
                        x, y, w, h = lrtb2xywh(1 , right - nwidth*2, top, bottom, CutImg3.width, CutImg3.height)
                        coords_flags[2].append([class_index, (x, y, w, h)])
                if nwidth*2 < left:#标签在图3上
                    x, y, w, h = lrtb2xywh(left-nwidth*2, right - nwidth * 2, top, bottom, CutImg3.width, CutImg3.height)
                    coords_flags[2].append([class_index, (x, y, w, h)])
            # 转换坐标并保存分割后的YOLO标签
            for i, coords_flag in enumerate(coords_flags):
                for class_index, (x, y, w, h) in coords_flag:
                    # 保存标签，确保类别ID为整数
                    with open(os.path.join(labsavepath, f'{fileName.split(".")[0]}CutImg{i + 1}.txt'), 'a') as f:
                        f.write(f'{int(class_index)} {x} {y} {w} {h}\n')


if __name__ == "__main__":
    Img1To3()




