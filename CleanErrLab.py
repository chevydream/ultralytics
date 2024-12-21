
import numpy as np
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os, random, shutil, torch

from cleanlab.classification import CleanLearning

k_fold_img_tar_Dir = None
k_fold_txt_tar_Dir = None

def copy_img_File(img0_file_Dir, txt0_file_Dir, img1_file_Dir, txt1_file_Dir):
    label0_pathDir = os.listdir(img0_file_Dir)  # 取图片的原始路径
    label1_pathDir = os.listdir(img1_file_Dir)  # 取图片的原始路径
    # filenumber=len(pathDir)
    # rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    label0_picknumber = 30
    label1_picknumber = 6
    sample_label0 = random.sample(label0_pathDir, label0_picknumber)  # 随机选取picknumber数量的样本图片
    sample_label1 = random.sample(label1_pathDir, label1_picknumber)  # 随机选取picknumber数量的样本图片
    # print (sample)
    for name in sample_label0:
        shutil.copy(img0_file_Dir + name, k_fold_img_tar_Dir + name)
        os.remove(img0_file_Dir + name)
        shutil.copy(txt0_file_Dir + name[0:7] + '.txt', k_fold_txt_tar_Dir + name[0:7] + '.txt')
        os.remove(txt0_file_Dir + name[0:7] + '.txt')
    for name in sample_label1:
        shutil.copy(img1_file_Dir + name, k_fold_img_tar_Dir + name)
        os.remove(img1_file_Dir + name)
        shutil.copy(txt1_file_Dir + name[0:7] + '.txt', k_fold_txt_tar_Dir + name[0:7] + '.txt')
        os.remove(txt1_file_Dir + name[0:7] + '.txt')
    return

def plot_examples(id_iter, nrows=1, ncols=1):
    plt.figure(figsize=(12, 8))
    for count, id in enumerate(id_iter):
        plt.subplot(nrows, ncols, count + 1)
        plt.imshow(X[id].reshape(28, 28), cmap="gray")
        plt.title(f"id: {id} \n label: {y[id]}")
        plt.axis("off")
    plt.tight_layout(h_pad=5.0)


from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, img_dir, label_dir):
        # 根文件路径
        self.root_dir = root_dir
        # 图片文件路径
        self.img_dir = img_dir
        # 标签文件夹路径
        self.label_dir = label_dir
        # 获取图片文件夹路径并生成图片名称的列表
        self.img_path = os.path.join(self.root_dir, self.img_dir)
        self.img_list = os.listdir(self.img_path)
        # 获取标签文件夹路径并生成标签名称的列表
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.label_list = os.listdir(self.label_path)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_item_path = os.path.join(self.img_path, img_name)
        # 读取对应路径的图片内容，生成图片对象，存储在img中
        img = Image.open(img_item_path)

        label_name = self.label_list[item]
        label_item_path = os.path.join(self.label_path, label_name)
        # 打开对应路径的txt文件，读取对应内容，存储在label中
        # file1 = open(label_item_path, "r")
        # label = file1.readline()
        boxes = torch.from_numpy(np.loadtxt(label_item_path).reshape(-1, 5))
        label = boxes[:, 0]

        return img, label

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    SEED = 123
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    warnings.filterwarnings("ignore", "Lazy modules are a new feature.*")


    # 导入数据集
    root_dir = "E:/DLDataSets/G3GQ-Small"
    img_dir = "images/val"
    label_dir = "labels/val"
    datasets = MyDataset(root_dir, img_dir, label_dir)

    all_images, all_labels = [], []
    for img, label in datasets:
        img_array = np.array(img) / 255.0
        all_images.append(img_array)
        all_labels.append(label)
        print(len(all_labels))
    # 将列表转换为numpy数组
    X = np.array(all_images).astype("float32")  # 二维数组
    y = np.array(all_labels).astype("int64")  # 一维标签
    print(X.shape, y.shape)

    # X = datasets["data"].astype("float32") # 二维数组
    # X /= 255.0  # 将图片像素值归一化到0~1
    # y = datasets["label"].astype("int64")  # 一维标签
    # print(X.shape, y.shape)

    # 训练YOLO模型
    #model = YOLO("E:/DLCode/weights/yolov8s-cls.pt")
    #model = YOLO("ultralytics/cfg/models/v8/myModel_yolov8-cls.yaml")
    #model.train(data="E:/DLDataSets/mnist", epochs=50, imgsz=28)

    model = YOLO("E:\\DLRuns\\detectV8\\train20240527_154850\\weights\\best.pt")

    # K重交叉验证
    num_crossval_folds = 3
    pred_probs = cross_val_predict(model, X, y, cv=num_crossval_folds, method="predict_proba")

    # 交叉训练的整体精度
    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(y, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc}")

    # 通过cleanlab库寻找噪声标签
    ranked_label_issues = CleanLearning(model).find_label_issues(X, y)
    #可以通过输入filter_by参数选择筛选方法，默认选择的是方法一，其他一些细节也可以进行调整
    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")

    # 对一些结果进行可视化
    plot_examples(ranked_label_issues[range(50)], 5, 10)
    clean_X = np.delete(X, list(ranked_label_issues), 0)
    clean_y = np.delete(y, list(ranked_label_issues), 0)
    print(clean_X.shape, clean_y.shape)
    clean_pred_probs = cross_val_predict(model, clean_X, clean_y, cv=num_crossval_folds, method="predict_proba")
    clean_predicted_labels = clean_pred_probs.argmax(axis=1)
    clean_acc = accuracy_score(clean_y, clean_predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {clean_acc}")



    # img0_file_Dir = "./ORimg0/"  # 源图片文件夹路径
    # txt0_file_Dir = "./ORtxt0/"
    # img1_file_Dir = "./ORimg1/"  # 源图片文件夹路径
    # txt1_file_Dir = "./ORtxt1/"
    # for i in range(10):
    #     os.mkdir('./Tar_img{}'.format(i))
    #     os.mkdir('./Tar_txt{}'.format(i))
    #     k_fold_img_tar_Dir = './Tar_img{}/'.format(i)  # 移动到新的文件夹路径
    #     k_fold_txt_tar_Dir = './Tar_txt{}/'.format(i)  # 移动到新的文件夹路径
    #     copy_img_File(img0_file_Dir, txt0_file_Dir, img1_file_Dir, txt1_file_Dir)

    # health = overall_label_health_score(labels, pred_probs)

