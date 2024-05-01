# deep_learning_Modular

## 实验前置条件：dataset

从```https://github.com/zalandoresearch/fashion-mnist```下载"t10k-images-idx3-ubyte.gz"、"t10k-labels-idx1-ubyte.gz"、"train-images-idx3-ubyte.gz"、"train-labels-idx1-ubyte.gz"四个文件并置于data文件夹下。


## 模型

自定义隐藏层大小、激活函数类型：在初始参数中设置"hidden_size"和"lr_choices"，且初始参数保存到相应的结果路径中。模型主要包括前向计算函数forward、梯度反向传播计算backward、参数更新函数update_params。

## 训练

对于每个epoch，训练使用的SGD优化器提取batch计算梯度并将之反向传播。学习率下降可以采取'exponential'、'step'、'cosine'三种方式。损失函数结合交叉熵损失和L2正则化损失，并且以l2_reg参数权衡两者的惩罚力度。每一次训练，选择所有epoch中在验证集上具有最好准确率，即结果最优，的模型权重。

#### 具体操作

1. 输入：改变train.py的参数
2. 运行如下代码：

```python train.py```

3. 输出：路径在'./outcome/train_outcome'，结果包括此次训练的最佳参数best_model_params.npz、plot图像、模型网络参数可视化图像。

## 测试

#### 具体操作

1. 输入：改变test.py的参数，载入想要的参数npz文件
2. 运行如下代码：

```python test.py```

3. 输出：路径在'./outcome/test_outcome'，结果包括此次网络参数可视化图像，并print出此次test的准确率。

## 参数查找

对隐藏层大小hidden_size、学习率lr、正则化强度l2、学习率调节参数drop的组合依次做训练和验证，也可以对学习率调节策略、其他学习率参数、batch_size、激活函数类型取不同的值，观察并记录了模型在不同超参数下的性能。

#### 具体操作

1. 输入：填写需要观察性能的参数
2. 运行如下代码：

```python search_best_params.py```

3. 输出：每个参数组合会在'./outcome/search_outcome'下创建'hidden_{hidden_size}_lr_{lr}_drop_{drop}_l2_{l2}'文件夹，并在此文件夹输出此组合的最佳参数文件best_model_params.npz、plot图像、模型网络参数可视化图像。最终把所有超参数组合的验证准确率汇总成'search.json'文件。

## 训练好的模型权重

[https://drive.google.com/file/d/1F9-dtNiwm99ARfFug2sdTTY55qbYpdgl/view?usp=sharing](https://drive.google.com/file/d/1F9-dtNiwm99ARfFug2sdTTY55qbYpdgl/view?usp=sharing)

参数对应性能：val_accuracy:90.45%,test_accuracy: 90.27%

