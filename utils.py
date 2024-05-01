import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def total_loss(Y_pred, Y_true,parameters, l2_reg):
    loss=cross_entropy(Y_pred, Y_true)+l2_loss(parameters, l2_reg)
    return loss
    
def cross_entropy(Y_pred, Y_true):
    m = Y_pred.shape[1] #64
    logprobs = -np.sum(Y_true * np.log(Y_pred + 1e-8))  # Adding a small epsilon to avoid log(0)
    loss = logprobs / m
    return loss

def l2_loss(parameters, l2_reg):
    l2_norm_squared = 0
    for param in parameters.values():
        if 'W' in param:
            l2_norm_squared += np.sum(np.square(param))
    return 0.5 * l2_reg * l2_norm_squared

def apply_l2_regularization(grads, parameters, l2_reg, m):
    for key in parameters.keys():
        grads["d"+key] += (l2_reg / m) * parameters[key]
    return grads

def compute_train_accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_true, axis=0)
    accuracy = np.mean(predictions == true_labels)
    return accuracy

def compute_val_accuracy(model, X_val, Y_val):
    Y_pred, _ = model.forward(X_val)
    predictions = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_val, axis=0)
    accuracy = np.mean(predictions == true_labels)
    return accuracy

def save_model_params(params, file_path):
    np.savez(file_path, **params)

def plot_loss(loss_list, result_dir):
    loss_train = loss_list['loss_train']
    loss_val = loss_list['loss_val']
    val_accuracy = loss_list['val_accuracy']
    length = len(loss_train) + 1
    epochs = range(1, length)

    # 设置 seaborn 风格
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 5))
    
    # 绘制训练和验证损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, label='Train Loss', color='blue')
    plt.plot(epochs, loss_val, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_plot.png'))
    plt.close()


def visualize_fc_weights(weights, save_dir):
    for i, W in weights.items(): #（64，784）
        if 'W' in i:

            # Get the shape of weight matrices
            num_neurons_prev_layer, num_neurons_curr_layer = W.shape #64
            num_rows = int(num_neurons_prev_layer ** 0.5) #16 横着摆这么多行
            num_cols = num_neurons_prev_layer // num_rows + 1 #17 

            plt.figure(figsize=(12, 8))

            # Loop through each neuron and plot its weights
            for j in range(num_neurons_prev_layer):  #64
                plt.subplot(num_rows, num_cols, j+1)
                plt.imshow(W[j, :].reshape((int(num_neurons_curr_layer ** 0.5), -1)))  #按照784进行分辨率划分
                plt.axis('off')

            # Save the figure
            plt.savefig(os.path.join(save_dir, f'weights_{i}.png'))

            # Close the figure to avoid memory leaks
            plt.close()

