import json
import os
import numpy as np
from lr_setting import LR_choice
from model import DeepNet
from utils import total_loss,cross_entropy, l2_loss, apply_l2_regularization, save_model_params, plot_loss
from utils import compute_train_accuracy,visualize_fc_weights
from load_mnist_data import load_data, one_hot_process

def train(model,X_train,Y_train,X_val,Y_val,epochs,initial_lr,batch_size,lr_choices,l2_reg=0.0,result_dir='outcome',decay_rate=0.95,epochs_drop=10,drop=0.5,milestones=[30,60]):
    ##初始化
    val_best=float('inf')
    best_params={}
    loss_list={'loss_train':[],'loss_val':[],'val_accuracy':[]}
    ##learning_date迭代方式
    lr_choice=LR_choice(initial_lr,lr_choices=lr_choices,decay_rate=decay_rate,epochs_drop=epochs_drop,drop=drop,milestones=milestones)

    for epoch in range(epochs):
        ## 先shuffle
        permutation=np.random.permutation(X_train.shape[1])
        X_shuffled=X_train[:,permutation]
        Y_shuffled=Y_train[:,permutation]
        ##更新learning_rate
        learning_rate=lr_choice.get_lr(epoch,epochs)
        ##初始化
        loss_train=0
        batches=0
        length=X_train.shape[1]

        for i in range(0,length,batch_size):
            ##取batch
            end=i+batch_size
            X_batch=X_shuffled[:,i:end]
            Y_batch=Y_shuffled[:,i:end]
            
            ##前向
            Y_pred,caches=model.forward(X_batch)
            
            ##算损失
            loss=total_loss(Y_pred,Y_batch,model.parameters,l2_reg)

            ##算梯度
            gradients=model.backward(X_batch,Y_batch,caches,l2_reg,batch_size)

            ##根据梯度更新参数
            model.update_params(gradients,learning_rate)

            loss_train+=loss
            batches+=1

        loss_train=loss_train/batches


        #validate
        ##前向
        Y_val_pred, _=model.forward(X_val)
        ##损失
        loss_val=total_loss(Y_val_pred,Y_val,model.parameters,l2_reg)
        ##准确率
        val_accuracy=compute_train_accuracy(Y_val_pred,Y_val)
        ##储存loss值
        loss_list['loss_train'].append(loss_train)
        loss_list['loss_val'].append(loss_val)
        loss_list['val_accuracy'].append(val_accuracy)

        ##保存某个epoch下的最佳参数
        if loss_val< val_best:
            best_val_loss=loss_val
            best_params={k:v.copy() for k,v in model.parameters.items()}
            ##保存最佳参数记录
            save_model_params(best_params,os.path.join(result_dir,'best_model_params.npz'))
        ##log
        print(f"Epoch{epoch+1}/Epoch_num{epochs},Train loss: {loss_train:.4f},Valid Accuracy:{val_accuracy:.2%},lr:{learning_rate:.4f}")


    model.parameters=best_params
    try:
        visualize_fc_weights(model.parameters,result_dir)
    except:
        pass
    
    return model,loss_list

def main():
    ##初始参数
    hidden_size=256
    milestones=[30,60]
    initial_lr=0.1
    batch_size=64
    l2_reg=0.1
    epochs=100
    decay_rate=0.95
    drop=0.2
    epochs_drop=10
    lr_choices='step'
    activation_type='relu'

    ##构建数据
    X_train,Y_train,X_val,Y_val,X_test,Y_test=load_data('./data')
    X_train,Y_train=one_hot_process(X_train,Y_train)
    X_val,Y_val=one_hot_process(X_val,Y_val)
    ##结果路径
    result_dir='./outcome/train_outcome'
    os.makedirs(result_dir,exist_ok=True)
    ##构建模型
    model=DeepNet([784,int(hidden_size),10],activation_type=activation_type)
    ##传入训练，
    trained_model,loss_list=train(model,X_train,Y_train,X_val,Y_val,epochs=epochs,initial_lr=initial_lr,batch_size=batch_size,lr_choices=lr_choices,
                                l2_reg=l2_reg,result_dir=result_dir,decay_rate=decay_rate,epochs_drop=epochs_drop,drop=drop,milestones=milestones)
    ##绘制结果
    plot_loss(loss_list,result_dir)


if __name__=="__main__":
    main()



