---
{"time":"2023/07/27","类型":null,"aliases":null,"genre":"目录","tags":null,"key":null,"dg-publish":true,"permalink":"/3 项目/Datawhale/Datawhale/","dgPassFrontmatter":true,"noteIcon":""}
---


# 锂离子电池生产参数调控及生产温度预测挑战赛

[2023 iFLYTEK A.I.开发者大赛-讯飞开放平台](https://challenge.xfyun.cn/topic/info?type=lithium-ion-battery&ch=ymfk4uU)

## 赛题解读

- 提供数据： 
	- 电炉上/下部各17组加热棒的**设定温度**
	- 电炉**底部17组**进气口的设定**进气流量**
- 任务：
	- 根据提供的数据样本**构建模型**，**预测**电炉上下部空间17个测温点的**测量温度值**

## 代码讲解

### 导包

```Python
# 更新lightGBM库&解压缩数据
# 注: 只需运行该行一次，若后续想要run all, 需要注释掉 !unzip data/data227148/data.zip, python的注释符号为 #
!pip install -U lightgbm
!unzip data/data227148/data.zip
```

> [!note]
> [[3 项目/Datawhale/lightGBM的数学原理\|lightGBM的数学原理]]

```Python
# 导入所需要的库
import pandas as pd # 用于处理数据的工具
import lightgbm as lgb # 机器学习模型 LightGBM
from sklearn.metrics import mean_absolute_error # 评分 MAE 的计算函数
from sklearn.model_selection import train_test_split # 拆分训练集与验证集工具
from tqdm import tqdm # 显示循环的进度条工具
```

### 数据准备

```Python
# 数据准备
train_dataset = pd.read_csv("./data/train.csv") # 原始训练数据。
test_dataset = pd.read_csv("./data/test.csv") # 原始测试数据（用于提交）。
submit = pd.DataFrame() # 定义提交的最终数据。创建一个空的数据级
submit["序号"] = test_dataset["序号"] # 对齐测试数据的序号。
MAE_scores = dict() # 定义评分项。
```

> [!note]
> 若为本地运行，应注意python的转义字符，例如改为`train_dataset = pd.read_csv("c:\\User\\train.csv")`,**注意为双斜杠**。如果不想加双斜杠的话，也可以在前面加r：`train_dataset = pd.read_csv(r"c:\User\train.csv")`

### 模型参数设置

**调整参数是优化模型性能的重要手段**
```Python
# 参数设置
pred_labels = list(train_dataset.columns[-34:]) # 需要预测的标签(训练数据计的最后34列）。
train_set, valid_set = train_test_split(train_dataset, test_size=0.2) # 拆分数据集。拆分比例为80%和20%
# 设定 LightGBM 训练参，查阅参数意义：https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgb_params = {
        'boosting_type': 'gbdt', #使用的提升方法，使用梯度提升决策树gbdt
        'objective': 'regression', #优化目标，这里设置为'regression',表示使用回归任务进行优化。
        'metric': 'mae',#评估指标，使用MAE，表示使用平均绝对误差作为评估指标。
        'min_child_weight': 5, #子节点中样本权重最小和，用于控制过拟合
        'num_leaves': 2 ** 5, #每棵树上的叶子节点树，影响模型的复杂度
        'lambda_l2': 10, # L2正则化项的权重，用于控制模型的复杂度
        'feature_fraction': 0.8, #随机选择特征的比例，用于防止过拟合
        'bagging_fraction': 0.8, #随机选择数据的比例，用于防止过拟合
        'bagging_freq': 4, #随机选择数据的频率，用于防止过拟合
        'learning_rate': 0.05, #学习率，控制每次迭代的步长
        'seed': 2023, #随机种子，用于产生随机性，保持结果的可重复性
        'nthread' : 16, #并行线程数，用于加速模型训练
        'verbose' : -1, #控制训练日志输出，-1表示禁用输出
    }
no_info = lgb.callback.log_evaluation(period=-1) # 禁用训练日志输出。
#LightGBM通常会输出一些训练过程的信息，通过回调函数可以避免输出这些信息，使得训练过程更简洁。
```

### 特征提取

```Python
# 时间特征函数

def time_feature(data: pd.DataFrame, pred_labels: list=None) -> pd.DataFrame:

    """提取数据中的时间特征。
    输入:
        data: Pandas.DataFrame
            需要提取时间特征的数据。
        pred_labels: list, 默认值: None
            需要预测的标签的列表。如果是测试集，不需要填入。
    输出: data: Pandas.DataFrame
            提取时间特征后的数据。
    """
    data = data.copy() # 复制数据，避免后续影响原始数据。
    data = data.drop(columns=["序号"]) # 去掉”序号“特征。
    data["时间"] = pd.to_datetime(data["时间"]) # 将”时间“特征的文本内容转换为 Pandas 可处理的格式。
    data["month"] = data["时间"].dt.month # 添加新特征“month”，代表”当前月份“。
    data["day"] = data["时间"].dt.day # 添加新特征“day”，代表”当前日期“。
    data["hour"] = data["时间"].dt.hour # 添加新特征“hour”，代表”当前小时“。
    data["minute"] = data["时间"].dt.minute # 添加新特征“minute”，代表”当前分钟“。
    data["weekofyear"] = data["时间"].dt.isocalendar().week.astype(int) # 添加新特征“weekofyear”，代表”当年第几周“，并转换成 int，否则 LightGBM 无法处理。
    data["dayofyear"] = data["时间"].dt.dayofyear # 添加新特征“dayofyear”，代表”当年第几日“。
    data["dayofweek"] = data["时间"].dt.dayofweek # 添加新特征“dayofweek”，代表”当周第几日“。
    data["is_weekend"] = data["时间"].dt.dayofweek // 6 # 添加新特征“is_weekend”，代表”是否是周末“，1 代表是周末，0 代表不是周末。
    data = data.drop(columns=["时间"]) # LightGBM 无法处理这个特征，它已体现在其他特征中，故丢弃。
    if pred_labels: # 如果提供了 pred_labels 参数，则执行该代码块。
        data = data.drop(columns=[*pred_labels]) # 去掉所有待预测的标签。
    return data # 返回最后处理的数据。
test_features = time_feature(test_dataset) # 处理测试集的时间特征，无需 pred_labels。
test_features.head(5)
```

### 训练和预测

```Python
# 从所有待预测特征中依次取出标签进行训练与预测。
for pred_label in tqdm(pred_labels):
    # print("当前的pred_label是：", pred_label)
    train_features = time_feature(train_set, pred_labels=pred_labels) # 处理训练集的时间特征。
    # train_features = enhancement(train_features_raw)
    train_labels = train_set[pred_label] # 训练集的标签数据。
    # print("当前的train_labels是：", train_labels)
    train_data = lgb.Dataset(train_features, label=train_labels) # 将训练集转换为 LightGBM 可处理的类型。
    valid_features = time_feature(valid_set, pred_labels=pred_labels) # 处理验证集的时间特征。
    # valid_features = enhancement(valid_features_raw)
    valid_labels = valid_set[pred_label] # 验证集的标签数据。
    # print("当前的valid_labels是：", valid_labels)
    valid_data = lgb.Dataset(valid_features, label=valid_labels) # 将验证集转换为 LightGBM 可处理的类型。
    # 训练模型，参数依次为：导入模型设定参数、导入训练集、设定模型迭代次数（200）、导入验证集、禁止输出日志
    model = lgb.train(lgb_params, train_data, 200, valid_sets=valid_data, callbacks=[no_info])
    valid_pred = model.predict(valid_features, num_iteration=model.best_iteration) # 选择效果最好的模型进行验证集预测。
    test_pred = model.predict(test_features, num_iteration=model.best_iteration) # 选择效果最好的模型进行测试集预测。
    MAE_score = mean_absolute_error(valid_pred, valid_labels) # 计算验证集预测数据与真实数据的 MAE。
    MAE_scores[pred_label] = MAE_score # 将对应标签的 MAE 值 存入评分项中。
    submit[pred_label] = test_pred # 将测试集预测数据存入最终提交数据中。
submit.to_csv('submit_result.csv', index=False) # 保存最后的预测结果到 submit_result.csv
```

> [!note]
> - `submit.to_csv('submit_result.csv', index=False)`中`index=False`是由于`pandas`导出的数据会新增一个标签（1，2，3，……），这个标签在赛题提交是不合规的，这里是为了去掉这个标签。

