
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from cleanlab.object_detection.filter import find_label_issues
from cleanlab.object_detection.rank import get_label_quality_scores
from cleanlab.object_detection.summary import visualize

import xml.etree.ElementTree as ET



def process_labels(label_paths, label_map):
    """
    处理标签文件并提取边界框、标签、忽略边界框和掩码。

    参数:
        label_paths (list): 标签文件路径的列表。
        label_map (dict): 标签名称到整数标签的映射。

    返回:
        list: 包含每个标签数据的字典列表。
    """
    labels = []

    for label_file in label_paths:
        # 解析标签文件
        label_tree = ET.parse(label_file)
        label_root = label_tree.getroot()

        # 提取真实标签的边界框、标签和掩码
        true_boxes = []
        true_labels = []
        masks = []
        bboxes_ignore = []

        for obj in label_root.findall('object'):
            # 提取边界框
            box = obj.find('bndbox')
            true_box = (float(box.find('xmin').text), float(box.find('ymin').text),
                        float(box.find('xmax').text), float(box.find('ymax').text))
            true_boxes.append(true_box)

            # 提取标签
            label_name = obj.find('name').text
            true_labels.append(label_map.get(label_name, -1))  # 使用 -1 表示未知标签

            # 提取掩码（如果存在）
            mask = obj.find('mask')
            if mask is not None:
                mask_points = [float(point.text) for point in mask.findall('point')]
                masks.append(mask_points)
            else:
                masks.append([])  # 没有掩码则添加空列表

            # 提取忽略边界框（如果存在）
            bndbox_ignore = obj.find('bndbox_ignore')
            if bndbox_ignore is not None:
                ignore_box = (float(bndbox_ignore.find('xmin').text), float(bndbox_ignore.find('ymin').text),
                              float(bndbox_ignore.find('xmax').text), float(bndbox_ignore.find('ymax').text))
                bboxes_ignore.append(ignore_box)
            else:
                bboxes_ignore.append([])  # 没有忽略边界框则添加空列表

        # 转换为 numpy 数组
        true_boxes = np.array(true_boxes, dtype=np.float32)
        true_labels = np.array(true_labels, dtype=np.int32)
        bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        # 创建 labels_data 字典形式
        labels_data = {
            'bboxes': true_boxes,  # 使用 numpy 数组
            'labels': true_labels,  # 使用 numpy 数组
            'bboxes_ignore': bboxes_ignore,  # 使用 numpy 数组
            'masks': masks,  # 使用 numpy 数组
            'seg_map': os.path.basename(label_file)  # 存储分割图的文件名
        }

        # 将标签数据追加到列表中
        labels.append(labels_data)

    return labels


def process_predictions(label_paths, prediction_dir, label_map):
    """
    处理预测文件并提取边界框和置信度，将每个类别的预测数据生成一个数组，并合并成列表。

    参数:
        label_paths (list): 标签文件路径的列表。
        prediction_dir (dict): 预测文件路径字典，键为标签文件名，值为预测文件路径。
        label_map (dict): 标签名称到类别编号的映射。

    返回:
        list: 每个标签文件生成的类别预测数组的列表，每个类别生成一个 (N, 5) 的数组。
    """
    all_predictions = []

    for label_file in label_paths:
        # 根据标签文件获取相应的预测文件
        pred_file_path = prediction_dir.get(os.path.basename(label_file))

        # 预创建每个类别的空预测数组，初始为 (0, 5)
        predictions_per_class = {label: np.empty((0, 5), dtype=np.float32) for label in label_map.values()}

        if pred_file_path and os.path.exists(pred_file_path):
            # 解析预测文件
            pred_tree = ET.parse(pred_file_path)
            pred_root = pred_tree.getroot()

            # 提取预测边界框和置信度
            for obj in pred_root.findall('object'):
                label_name = obj.find('name').text
                if label_name in label_map:
                    # 取得该类别对应的标签编号
                    label_idx = label_map[label_name]

                    box = obj.find('bndbox')
                    confidence = float(obj.find('confidence').text)
                    pred_box = (float(box.find('xmin').text), float(box.find('ymin').text),
                                float(box.find('xmax').text), float(box.find('ymax').text))

                    # 将预测框和置信度合并为一个列表，追加到对应类别的数组中
                    predictions_per_class[label_idx] = np.vstack(
                        [predictions_per_class[label_idx], np.array([*pred_box, confidence], dtype=np.float32)]
                    )

        # 按类别顺序生成最终的预测结果列表
        predictions_data = [predictions_per_class[label] for label in range(len(label_map))]
        all_predictions.append(predictions_data)

    return all_predictions


def example():
    # 定义图像路径
    IMAGE_PATH = 'E:/DLCode/cleanlab/example_images/1/'

    # 加载标签和预测
    predictions = pickle.load(open("E:/DLCode/cleanlab/1/predictions.pkl", "rb"))
    labels = pickle.load(open("E:/DLCode/cleanlab/1/labels.pkl", "rb"))

    print("labels数:", len(labels))
    print("predictions数:", len(predictions))

    # 查找标签问题的图像索引
    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)

    # 显示具有最严重标签问题的图像索引
    num_examples_to_show = 5
    print("具有标签问题的图像索引：", label_issue_idx[:num_examples_to_show])

    # 获取每个图像的标签质量分数
    scores = get_label_quality_scores(labels, predictions)
    print("标签质量分数：", scores[:num_examples_to_show])

    # 根据分数阈值获取标签问题图像索引
    from cleanlab.object_detection.rank import issues_from_scores

    issue_idx = issues_from_scores(scores, threshold=0.5)
    print("根据分数阈值获取的标签问题图像索引：", issue_idx[:num_examples_to_show])
    print("这些问题图像的分数：", scores[issue_idx][:num_examples_to_show])


    # 5. 可视化标签问题
    def visualize_issue(index):
        image_path = IMAGE_PATH + labels[index]['seg_map']
        label = labels[index]
        prediction = predictions[index]
        score = scores[index]

        print(f"图像路径：{image_path} | 索引：{index} | 标签质量分数：{score} | 是否存在问题：是")
        visualize(image_path, label=label, prediction=prediction, overlay=False)


    # 示例可视化标签问题图像
    visualize_issue(issue_idx[0])  # 更改为其他图像索引以查看不同图像
    visualize_issue(issue_idx[1])


def issues_from_scores(scores, threshold):
    """
    根据评分列表生成需要关注的问题列表。

    参数:
    scores (list of float): 评分列表。
    threshold (float): 阈值，低于此值的评分将被视为问题。

    返回:
    list of int: 需要关注的问题索引列表。
    """
    issues = []
    for index, score in enumerate(scores):
        if score < threshold:
            issues.append(index)
    return issues


def yoloTxt(file_path):
    base_dir = 'E:/DLCode/cleanlab/2/'

    def read_paths(file_path):
        """从文件中读取路径列表"""
        with open(file_path, 'r') as f:
            paths = [line.strip() for line in f]
        return paths

    # 定义路径
    prediction_paths = read_paths(os.path.join(base_dir, 'predpath.txt'))
    label_paths = read_paths(os.path.join(base_dir, 'labelpath.txt'))

    # 从文件中加载预测和标签
    prediction_dir = {os.path.basename(p): p for p in prediction_paths}
    label_map = {'thyroid_nodule': 0}  # 标签名称到整数标签的映射

    # 处理标签和预测
    labels = process_labels(label_paths, label_map)
    predictions = process_predictions(label_paths, prediction_dir, label_map)

    print("labels数:",len(labels))
    print("predictions数:",len(predictions))

    # 查找标签问题的图像索引
    label_issue_idx = find_label_issues(labels, predictions, return_indices_ranked_by_score=True)
    num_examples_to_show = 5
    print("具有标签问题的图像索引：", label_issue_idx[:num_examples_to_show])

    # 获取每个图像的标签质量分数
    scores = get_label_quality_scores(labels, predictions)
    print("标签质量分数：", scores[:num_examples_to_show])

    # 根据分数阈值获取标签问题图像索引
    issue_idx = issues_from_scores(scores, threshold=0.5)
    print("根据分数阈值获取的标签问题图像索引：", issue_idx[:num_examples_to_show])
    print("这些问题图像的分数：", scores[issue_idx][:num_examples_to_show])


    def save_low_score_xml_paths(xml_paths, scores, threshold, output_file):
        """
        保存分数低于阈值的标签文件名称（不含扩展名）到一个 TXT 文件中。

        参数:
            xml_paths (list): 原始的 XML 文件路径列表。
            scores (list): 每个图像的标签质量分数列表。
            threshold (float): 标签质量分数的阈值，小于该值的图像将被保存。
            output_file (str): 保存路径的文件名。
        """
        low_score_filenames = []

        for i, score in enumerate(scores):
            if score < threshold:
                # 获取文件名并去除 .xml 扩展名
                filename = os.path.splitext(os.path.basename(xml_paths[i]))[0]
                low_score_filenames.append(filename)

        with open(output_file, 'w') as f:
            for filename in low_score_filenames:
                f.write(filename + '\n')

        print(f"已将分数低于 {threshold} 的 {len(low_score_filenames)} 个图像名称保存到 {output_file}。")


    # 设置保存路径和分数阈值
    output_file = os.path.join(base_dir, 'E:/DLCode/cleanlab/2/low_score_paths.txt')
    score_threshold = 0.5

    # 保存分数低的图像的 XML 文件路径
    save_low_score_xml_paths(label_paths, scores, score_threshold, output_file)



if __name__ == "__main__":
    #example()

    yoloTxt()


