import numpy as np
import os
from utils import plot_loss
import json
from model import DeepNet
from train import train 
from load_mnist_data import one_hot_process, load_data
from utils import compute_val_accuracy


def search(data_train, data_val):
    ##初始
    lrs =[0.1,0.01,0.001,0.0001] #0.1还可以
    lr_choices = 'step'
    epochs_drop = 10
    drops = [0.2,0.4,0.6,0.8]  #0.2、0.8还可以
    milestones =[30,60]
    decay_rate = 0.8
    hidden_sizes = [64,128,256,512]
    l2_regs = [0.0,0.01,0.001,0.0001]
    activation_type = 'relu'  
    epochs = 100
    batch_size = 64
    val_best = 0
    best_params = None
    #search_results =[]
    search_results ={}


    ##train和val结果路径
    results_dir='./outcome/search_outcome/' 
    os.makedirs(results_dir,exist_ok=True)
    
    ##参数组合,训练过程中最佳参数保存在results_dir下
    for hidden_size in hidden_sizes:  
        for lr in lrs:
            for drop in drops:
                    for l2 in l2_regs:

                        ##搜索结果路径
                        search_dir = f'hidden_{hidden_size}_lr_{lr}_drop_{drop}_l2_{l2}'  #这是某个参数条件的输出结果文件夹名称
                        os.makedirs(os.path.join(results_dir,search_dir),exist_ok=True)
                        search_dir=results_dir+search_dir

                        ##构建模型
                        model = DeepNet([784, hidden_size, 10])
                        ##训练       
                        trained_model, loss_list = train(model,data_train[0], data_train[1],data_val[0], data_val[1],epochs = epochs,
                                                            initial_lr =lr,batch_size = batch_size,lr_choices = lr_choices,
                                l2_reg = l2,result_dir = search_dir,decay_rate = decay_rate,epochs_drop = epochs_drop,
                                                    drop = drop,milestones = milestones)
                        
                        ##绘制结果
                        plot_loss(loss_list, search_dir)

                        ##val
                        ###计算
                        val_accuracy = compute_val_accuracy(trained_model, data_val[0], data_val[1])  #
                        ### 搜索记录
                        search_results.update({f"choices_{len(search_results)}": {"hidden_size": hidden_size, "lr": lr, "l2": l2, "drop": drop, "val_accuracy": val_accuracy}})

    return search_results

def main():
    ##数据
    X_train,Y_train, X_val, Y_val,X_test,Y_test = load_data('./data', training=True, val_split=0.1)
    X_train, Y_train = one_hot_process(X_train, Y_train)
    X_val, Y_val= one_hot_process(X_val, Y_val)
    ##进行搜索
    search_results = search((X_train, Y_train), (X_val, Y_val))

    with open('search.json', 'w')as file:
        json.dump(search_results,file,indent=4)

if __name__=='__main__':
    main()

