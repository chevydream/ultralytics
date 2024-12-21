# ultralytics修改与使用说明

## 一.ultralytics代码修改

从官网下载ultralytics代码，并对其做修改，以满足以下需求：
- 1 训练：模型训练, 模型导出
- 2 测试：识别率测试
- 3 导出：dnn用的onnx模型，rknn用的rknnopt和onnx模型，VINO用的IR模型。

### 1.1.为测试而做的修改 
- 1 default.yaml  
- 2 loaders.py     解决加载图片时的中文路径乱码问题  
- 3 utils.py       训练时, 按指定要求过滤要加载样本  
- 4 predictor.py   解决保存图片时的中文路径乱码问题  
- 5 results.py     解决识别为多个类别时的标签覆盖问题  
- 6 plotting.py    解决识别为多个类别时的标签覆盖问题 && 在直方图柱子的顶部添加计数标注  
- 7 patches.py     让opencv对话框不要自适应图片大小  
- 8 trainer.py     若已经训练完成了, 还想继续训练时, 需要有这句才行!!!  

### 1.2.为rknn而做的修改
- 1.default.yaml
- 2.exporter.py
- 3.head.py
- 4.autobackend.py
- 5.augment.py
- 6.torch_utils.py

### 1.3.为VINO而做的修改


### 1.4.自己写的测试文件
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

5.说明文件
- 1.README.md

## 二.Yolov8模型转RKNPU模型说明

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


## 三. Yolo11模型转RKNPU模型说明

## Source

​	本仓库基于 https://github.com/ultralytics/ultralytics  仓库的 50497218c25682458ea35b02dcc5d8a364f34591 commit 进行修改,验证.



## 模型差异

在基于不影响输出结果, 不需要重新训练模型的条件下, 有以下改动:

- 修改输出结构, 移除后处理结构. (后处理结果对于量化不友好)

- dfl 结构在 NPU 处理上性能不佳，移至模型外部的后处理阶段，此操作大部分情况下可提升推理性能。


- 模型输出分支新增置信度的总和，用于后处理阶段加速阈值筛选。 


以上移除的操作, 均需要在外部使用CPU进行相应的处理. (对应的后处理代码可以在 **RKNN_Model_Zoo** 中找到)



## 导出onnx模型

在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
# 调整 ./ultralytics/cfg/default.yaml 中 model 文件路径，默认为 yolo11n.pt，若自己训练模型，请调接至对应的路径。支持检测、分割、姿态、旋转框检测模型。
# 如填入 yolo11n.pt 导出检测模型
# 如填入 yolo11n-seg.pt 导出分割模型
# 如填入 yolo11n-pose.pt 导出姿态模型
# 如填入 yolo11n-obb.pt 导出OBB模型

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# 执行完毕后，会生成 ONNX 模型. 假如原始模型为 yolo11n.pt，则生成 yolo11n.onnx 模型。
```



## 转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo

