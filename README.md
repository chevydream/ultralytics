# ultralytics修改与使用说明

## 一.ultralytics代码修改

从官网下载ultralytics-yolov8代码，并对其做修改，以满足以下需求：
- 1）训练：日常的模型训练, 模型导出等
- 2）测试：日常识别率测试
- 3）导出：dnn的onnx模型，rknn的rknnopt模型，VINO的IR模型。

### 8.2.1.为测试而做的修改 
- 1 default.yaml  
- 2 loaders.py  
- 3 utils.py  
- 4 predictor.py  
- 5 results.py  
- 6 val.py  
- 7 plotting.py  
- 8 patches.py  

### 8.2.2.为rknn而做的修改
- 1.default.yaml
- 2.exporter.py
- 3.head.py
- 4.autobackend.py

### 8.2.3.为VINO而做的修改
- 1.exporter.py  
说明: 新版ultralytics对torch和openvino的版本要求太高了, 5.28是win7系统, 不支持太高的版本, 故把ultralytics8.1.0上的老函数又找回来了

### 8.2.4.自己写的测试文件
1.数据集
- 1.myData_CarHead.yaml
- 2.myData_cpc.yaml
- 3.myData_face.yaml
- 4.myData_G3.yaml
- 5.myData_hsjlun.yaml
- 6.myData_IVAS.yaml
- 7.myData_RoadCars.yaml
- 8.myData_tempPlate.yaml	

2.模型结构
- myModel_yolov8-cls.yaml
- myModel_yolov8s.yaml
- myModel_yolov8-p6.yaml

3.测试代码1 -Train, Predict, Export
- CleanErrLab.py
- demo_Img1To3.py
- demo_Pic BiaoDing.py
- demo_Pic FenLei.py
- demo_Pic Veh2Lun.py
- demo_Train.py
- demo_Video.py

4.测试代码2 -VINO相关	
- demo_VINO.py
- demo_VINO_lh.py
- demo_VINO_pg.py
- demo_VINO_tl.py
- 
5.说明文件
- 1.README.zh-MY.md

## 二.导出 RKNPU 模型说明

### 1.模型结构上的调整

- dfl 结构在 NPU 处理上性能不佳，移至模型外部。

  假设有6000个候选框，原模型将 dfl 结构放置于 ''框置信度过滤" 前，则 6000 个候选框都需要计算经过 dfl 计算；而将 dfl 结构放置于 ''框置信度过滤" 后，假设过滤后剩 100 个候选框，则dfl部分计算量减少至 100 个，大幅减少了计算资源、带宽资源的占用。



- 假设有 6000 个候选框，检测类别是 80 类，则阈值检索操作需要重复 6000* 80 ～= 4.8*10^5 次，占据了较多耗时。故导出模型时，在模型中额外新增了对 80 类检测目标进行求和操作，用于快速过滤置信度。(该结构在部分情况下对有效，与模型的训练结果有关)

  可以在 **./ultralytics/nn/modules/head.py**  52行～54行的位置，注释掉这部分优化，对应的代码是:

  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  y.append(cls_sum)
  ```




- (optional) 实际上，用户可以参考yolov5的结构，将80类输出调整为 80+1类，新增的1类作为控制框的置信度，起到快速过滤作用。这样后处理在cpu执行阈值判断的时候，就可以减少 10～40倍的逻辑判断次数。



### 2.导出模型操作

在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
# 调整 ./ultralytics/cfg/default.yaml 中 model 文件路径，默认为 yolov8n.pt，若自己训练模型，请调接至对应的路径

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

执行完毕后，会生成 _rknnopt.torchscript 模型。假如原始模型为 yolov8n.pt，则生成 yolov8n_rknnopt.torchscript 模型。
```



导出代码改动解释

- ./ultralytics/cfg/default.yaml 导出模型格式的参数 format, 添加了 'rknn' 的支持
- 模型推理到 Detect Head 时，format=='rknn'生效，跳过dfl与后处理，输出推理结果
- 需要注意，本仓库没有测试对 pose head, segment head 的优化方式，目前暂不支持，如果需求可尝试自行更改。



### 3.转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo 

