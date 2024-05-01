import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_fc2(parameters,save_dir):
    for key, value in parameters.items():
        if key.startswith("W2"):
            # Get the shape of weight matrices
            num_neurons_prev_layer, num_neurons_curr_layer = value.shape #10,512

            plt.figure(figsize=(12, 8))

            # Loop through each neuron and plot its weights
            # for j in range(num_neurons_prev_layer):  #遍历10张图
            #     plt.subplot(num_neurons_curr_layer, num_neurons_prev_layer, j+1)  #绘制每个子图的位置，横着是512，竖着是10，
            #     #plt.imshow(value[j, :].reshape((num_neurons_curr_layer,-1)))  #按照784进行分辨率划分
            #     #weight=np.repeat(value[j, :].reshape((512,1)), 512, axis=1)
            #     #print(weight.shape)
            #     plt.imshow(value[j, :].reshape((512,1)))
            #     #plt.imshow(weight)
            #     plt.axis('off')
            
            num_rows = int(num_neurons_prev_layer ** 0.5) #16 横着摆这么多行,3
            num_cols = num_neurons_prev_layer // num_rows + 1 #17 ,4

            for j in range(num_neurons_prev_layer):
                plt.subplot(num_rows, num_cols, j+1)
                weight=np.repeat(value[j, :].reshape((num_neurons_curr_layer,1)), num_neurons_curr_layer, axis=1)
                plt.imshow(weight)
                plt.axis('off')

            # Save the figure
            plt.savefig(os.path.join(save_dir, f'weights_{key}.png'))

            # Close the figure to avoid memory leaks
            plt.close()




def main():
    ##初始参数
    hidden_size=64
    milestones=[30,60]
    initial_lr=0.01
    batch_size=64
    l2_reg=0.01
    epochs=2
    decay_rate=0.95
    drop=0.6
    epochs_drop=10
    lr_choices='step'
    activation_type='relu'

    result_dir='./outcome/visualize/'
    params = np.load(f'./outcome/search_outcome_1/hidden_{hidden_size}_lr_{initial_lr}_drop_{drop}_l2_{l2_reg}/best_model_params.npz')
    os.makedirs(result_dir,exist_ok=True)

    # 获取权重参数
    W = params['W2']

    visualize_fc2(params,result_dir)

    


if __name__=="__main__":
    main()