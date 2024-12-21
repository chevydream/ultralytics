import shutil, time, os
from ultralytics import YOLO
from tkinter import filedialog
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model,save_model
import global_var

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#网络结构
NetStruct_Det_V8n   = "ultralytics/cfg/models_my/myModel_yolov8s.yaml"
NetStruct_Det_11n   = "ultralytics/cfg/models_my/myModel_yolo11n.yaml"
NetStruct_Cls       = "ultralytics/cfg/models_my/myModel_yolov8-cls.yaml"

#预训模型
PreModel_V8n        = "E:/DLModel/curWeights/yolov8n.pt"
PreModel_V8s        = "E:/DLModel/curWeights/yolov8s.pt"
PreModel_11n        = "E:/DLModel/curWeights/yolo11n.pt"
PreModelCls_n       = "E:/DLModel/curWeights/yolov8n-cls.pt"
PreModelCls_s       = "E:/DLModel/curWeights/yolov8s-cls.pt"

#训练数据
DataSets_CarHead    = "ultralytics/cfg/datasets_my/myData_CarHead.yaml"
DataSets_cpc        = "ultralytics/cfg/datasets_my/myData_cpc.yaml"
DataSets_G3Old      = "ultralytics/cfg/datasets_my/myData_G3Old.yaml"
DataSets_G3New      = "ultralytics/cfg/datasets_my/myData_G3New.yaml"
DataSets_G3Quan     = "ultralytics/cfg/datasets_my/myData_G3Quan.yaml"
DataSets_FaceLunHsj = "ultralytics/cfg/datasets_my/myData_FaceLunHsj.yaml"
DataSets_IVAS       = "ultralytics/cfg/datasets_my/myData_IVAS.yaml"
DataSets_tempPlate  = "ultralytics/cfg/datasets_my/myData_tempPlate.yaml"
DataSets_UCAS       = "ultralytics/cfg/datasets_my/myData_UCAS.yaml"
DataSets_UpDown     = "ultralytics/cfg/datasets_my/myData_UpDown.yaml"

DataSets_back       = "E:/DLDataSets/background_cls/"

# ======================================================================================================
#需配置的参数

#运行控制变量
runControl      = 2      # 运行模式：1-训练(结构)，2-训练(pt)，3-训练(续训)，4-导出（三种），5-导出（16位）

projectPath     = "E:/DLRuns/detectV11"

#三选一: 模型结构|预训练模型|续训练模型
NetStruct       = NetStruct_Det_11n
PreModel        = PreModel_11n
PreModel_jx     = "train20240625_190123"

onnxOpset       = 12    #yolov8|yolov11>=12, rknn<=12

#数据集
DataSets        = DataSets_FaceLunHsj

GuolvFlag       = 1     #加载哪些标签: 1-加载所有标签, 2-门架加载部分样本（车轮）标签, 3-门架加载样本(车型 + 危险品)标签
                        #4-G3加载部分样本（车轮）标签, 5-G3加载（车轮 + 后视镜）标签, 6-G3加载样本（车型 + 危险品）标签

#特别注意: 一定要记得清空标签缓存, 否则修改下面的过滤条件会不起作用

DaoChuFlag		= 3		#1-导出rknn用的Onnx模型, 2-导出rknn用的Opt模型, 3-都导出

#配置推荐: 250|1024|160; 250|128|320; 250|72|640;
myEpochs        = 250
myBatch         = 72
myImgsize       = 640

# ======================================================================================================


def myExport_fp16(saveName, trainFlag):

    model_path = filedialog.askopenfilename()

    if model_path is None:
        print("请选择一个模型文件(pt)，否则无法进行转换！")
        return

    onnx_model = load_model(model_path)
    fp16_model = convert_float_to_float16(onnx_model, keep_io_types=False)
    save_model(fp16_model, model_path.replace(".onnx", "-fp16.onnx"))


def myExport(saveName, trainFlag):

    if trainFlag == 0:
        model_path = filedialog.askopenfilename()
        if model_path is None:
            print("请选择一个模型文件(pt)，否则无法进行转换！")
            return
    else:
        model_path = "{:}/{:}/weights/best.pt".format(projectPath, saveName)

    # 加载已经存在的模型
    model = YOLO(model_path)

    # 说明：转换时，若不开启GPU，onnx的文件大小会大一倍，不知道为啥(现在都是开着转换的)

    # 将模型导出为标准的 ONNX 格式(fp32)
    model.export(format="onnx", half=False, dynamic=False, opset=onnxOpset, simplify=True)

    # 将模型导出为 OpenVINO 格式(fp16)      说明: 必须指定为CPU, 否则会报错, 转换不出来
    model.export(format="openvino", half=True, dynamic=False, opset=onnxOpset, simplify=True, device='CPU')

    # 将模型导出为用于rknn的 rknnopt 和 onnx 格式(fp16)
    model_path2 = model_path.replace(".pt", "-rknn.pt", 1)
    shutil.copy2(model_path, model_path2)
    model2 = YOLO(model_path2)
    model2.export(format="rknn", half=True, dynamic=False, opset=onnxOpset)
    os.remove(model_path2)

    # 将模型导出为标准的 ONNX 格式(fp16)
    model_path3 = model_path.replace(".pt", "-fp16.pt", 1)
    shutil.copy2(model_path, model_path3)
    model3 = YOLO(model_path3)
    model3.export(format="onnx", half=True, dynamic=False, opset=onnxOpset)
    os.remove(model_path3)


def myTrain(typeFlag, saveName):
    # 先必须在主模块初始化（只在Main模块需要一次即可）
    global_var._init()
    # 定义跨模块全局变量
    global_var.set_value('GuolvFlag', GuolvFlag)
    global_var.set_value('DaoChuFlag', DaoChuFlag)

    if typeFlag < 1 or typeFlag > 3:
        return

    # 从头开始构建新模型
    if typeFlag == 1:
        model = YOLO(NetStruct)
        model.train(data=DataSets, epochs=myEpochs, batch=myBatch, imgsz=myImgsize, name=saveName)

    # 加载预训练模型（推荐用）
    if typeFlag == 2:
        model = YOLO(PreModel)
        model.train(data=DataSets, epochs=myEpochs, batch=myBatch, imgsz=myImgsize, name=saveName)

    # 中断后继续之前的训练，请用这句
    if typeFlag == 3:
        model = YOLO("{:}/{:}/weights/best.pt".format(projectPath, PreModel_jx))
        # model.train(data=DataSets, resume=True) # 应该不用配置这三个参数了,需测试验证一下
        model.train(data=DataSets, epochs=myEpochs, batch=myBatch, imgsz=myImgsize, resume=True)

    # 将模型拷贝到当前目录，方便下次能自动加载最新模型
    srcPath = "{:}/{:}/weights/best.pt".format(projectPath, saveName)
    savePath = "E:/DLModel/curWeights/best.pt"
    shutil.copyfile(srcPath, savePath)



if __name__ == "__main__":
    if runControl != 3:
        curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        saveName = "train{:}".format(curtime)
    else:
        saveName = "train{:}".format(PreModel_jx)

    # 训练得到 pt模型，然后导出为 onnx模型 和 rknn模型 和 IR模型
    if runControl >= 1 and runControl <= 3:
        myTrain(runControl, saveName)
        myExport(saveName, 1)

    # 将pt模型 导出为 onnx模型 和 rknn模型 和 IR模型
    if runControl == 4:
        myExport(saveName, 0)

    # 将 32位 onnx模型 转换为 16位 onnx模型
    if runControl == 5:
        myExport_fp16(saveName, 0)



