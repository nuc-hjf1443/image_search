# 定义一个字典，保存分类号和中文名的映射关系
classification_names = {
    0: '上身衣服',
    1: '鞋',
    2: '包',
    3: '下身衣服',
    4: '手表',
}

# 数据文件路径和预处理配置
FASHION_LABELS_PATH = "../common/fashion-labels.csv"
IMG_PATH = '../common/dataset'
IMG_H = 64
IMG_W = 64

# 随机性和数据划分配置
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 训练超参数
LEARNING_RATE = 1e-3
EPOCHS = 20
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

# 包和模型文件定义
PACKAGE_NAME = "image_classification"
CLASSIFIER_MODEL_NAME = 'classifier.pt'