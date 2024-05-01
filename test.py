import json
import os
import numpy as np
from lr_setting import LR_choice
from model import DeepNet
from utils import total_loss,cross_entropy, l2_loss, apply_l2_regularization, save_model_params, plot_loss
from utils import compute_train_accuracy,visualize_fc_weights
from load_mnist_data import load_data, one_hot_process

def test(model,X_test,Y_test,l2_reg,hidden_size,lr,l2,drop):
    ##前向
    Y_test_pred, _=model.forward(X_test)
    ##损失
    loss_val=total_loss(Y_test_pred,Y_test,model.parameters,l2_reg)
    ##准确率
    test_accuracy=compute_train_accuracy(Y_test_pred,Y_test)
    ##log
    with open('test_output.txt', 'a') as f:
        f.write(f"test loss: {loss_val:.4f}, test Accuracy: {test_accuracy:.2%}, model params: hidden_size={hidden_size}, lr={lr}, drop={drop}, l2={l2_reg}\n")

    return test_accuracy

def main():

    with open('test_output.txt', 'w') as f:
        f.write(f'test accuracy\n')

    # 读取JSON文件
    with open('search.json', 'r') as f:
        data = json.load(f)
    filtered_params={}
    # 提取val_accuracy大于0.9的参数
    for i,param in data.items():
        if param['val_accuracy'] >= 0.9:
            filtered_params[i] = param
    

    for i,param in filtered_params.items():
        ##初始化参数
        hidden_size=param["hidden_size"]
        milestones=[30,60]
        initial_lr=param["lr"]
        batch_size=64
        l2_reg=param["l2"]
        decay_rate=0.8
        drop=param["drop"]
        epochs_drop=10
        lr_choices='step'
        activation_type='relu'

        ##构建数据
        X_train,Y_train,X_val,Y_val,X_test,Y_test=load_data('./data')
        X_test,Y_test=one_hot_process(X_test,Y_test)
        ##构建模型
        model=DeepNet([784,int(hidden_size),10],activation_type=activation_type)

        ##读取参数文件
        load_dir=f'hidden_{hidden_size}_lr_{initial_lr}_drop_{drop}_l2_{l2_reg}' 
        model.load_parameters(f"./outcome/search_outcome_1/{load_dir}/best_model_params.npz")
        test_accuracy=test(model,X_test,Y_test,l2_reg=l2_reg,hidden_size=hidden_size,lr=initial_lr,l2=l2_reg,drop=drop)

        ##训练好的模型网络参数的可视化
        test_dir='./outcome/test_outcome/'+load_dir
        os.makedirs(test_dir,exist_ok=True)
        try:
            visualize_fc_weights(model.parameters, test_dir)
        except:
            pass

if __name__=="__main__":
    main()