import nncf
from openvino.tools import mo
from openvino.runtime import serialize
from tqdm.notebook import tqdm
from ultralytics.utils.metrics import ConfusionMatrix
import torch
import numpy as np
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG


def my_test(model, core, data_loader:torch.utils.data.DataLoader, validator, num_samples:int = None):
    """
    OpenVINO YOLOv8 model accuracy validation function. Runs model validation on dataset and returns metrics
    Parameters:
        model (Model): OpenVINO model
        data_loader (torch.utils.data.DataLoader): dataset loader
        validato: instalce of validator class
        num_samples (int, *optional*, None): validate model only on specified number samples, if provided
    Returns:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
    """
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    num_outputs = len(model.outputs)
    compiled_model = core.compile_model(model)
    for batch_i, batch in enumerate(tqdm(data_loader, total=num_samples)):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [torch.from_numpy(results[compiled_model.output(0)]), torch.from_numpy(results[compiled_model.output(1)])]
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats


def print_stats(stats:np.ndarray, total_images:int, total_objects:int):
    """
    Helper function for printing accuracy statistic
    Parameters:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, key is metric name, value is metric value
        total_images (int) -  number of evaluated images
        total objects (int)
    Returns:
        None
    """
    # print("Boxes:")
    mp, mr, map50, mean_ap = stats['metrics/precision(B)'], stats['metrics/recall(B)'], stats['metrics/mAP50(B)'], stats['metrics/mAP50-95(B)']
    # Print results
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print('Boxes:',s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('\t\t\tall', total_images, total_objects, mp, mr, map50, mean_ap))
    if 'metrics/precision(M)' in stats:
        # print("Masks:")
        s_mp, s_mr, s_map50, s_mean_ap = stats['metrics/precision(M)'], stats['metrics/recall(M)'], stats['metrics/mAP50(M)'], stats['metrics/mAP50-95(M)']
        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
        print("Masks:",s)
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('\t\t\tall', total_images, total_objects, s_mp, s_mr, s_map50, s_mean_ap))

if __name__ == "__main__":
    #定义需要量化的onnx模型的路径以及量化后保存的名称
    MODEL_NAME = "yolov8n.onnx"
    MODEL_PATH = f"E:/DLModel/models"
    IR_MODEL_NAME = "yolov8n"

    #将需要量化的onnx模型路径串起来
    onnx_path = f"{MODEL_PATH}/{MODEL_NAME}"

    #定义量化后的模型的路径名称
    FP32_path = f"{MODEL_PATH}/FP32_openvino_model/{IR_MODEL_NAME}_FP32.xml"
    FP16_path = f"{MODEL_PATH}/FP16_openvino_model/{IR_MODEL_NAME}_FP16.xml"
    Int8_path = f"{MODEL_PATH}/Int8_openvino_model/{IR_MODEL_NAME}_Int8.xml"

    #FP32 model
    model = mo.convert_model(onnx_path)
    serialize(model, FP32_path)
    print(f"export ONNX to Openvino FP32 IR to:{FP32_path}")

    #FP16 model
    model = mo.convert_model(onnx_path, compress_to_fp16=True)
    serialize(model, FP16_path)
    print(f"export ONNX to Openvino FP16 IR to:{FP16_path}")

    #Int8 model

    def create_data_source():
        args = get_cfg(cfg=DEFAULT_CFG)
        args.data = "E:/DLDataSets/COCO/coco.yaml"
        det_validator = DetectionValidator(args=args)
        det_validator.data = check_det_dataset(args.data)
        det_dataloader = det_validator.get_dataloader("E:/DLDataSets/COCO", 1)
        return det_dataloader

    def transform_fn(data_item):
        images = data_item['img']
        images = images.float()
        images = images / 255.0
        images = images.cpu().detach().numpy()
        return images

    #加载数据
    data_source = create_data_source()

    #实例化校准数据集
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)

    #配置量化管道
    subset_size = 40
    preset = nncf.QuantizationPreset.MIXED

    #执行模型量化
    from openvino.runtime import Core
    from openvino.runtime import serialize

    core = Core()
    ov_model = core.read_model(FP16_path)
    quantized_model = nncf.quantize(
        ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size
    )
    serialize(quantized_model, Int8_path)
    print(f"export ONNX to Openvino Int8 IR to:{Int8_path}")

