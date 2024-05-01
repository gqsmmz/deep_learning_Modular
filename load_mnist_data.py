import numpy as np
import requests
import gzip
import os

def load_data(data_dir, training=True, val_split=0.2):
    files=["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]

    def load_images(file_name):
        with gzip.open(os.path.join(data_dir, file_name),'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,784)#
        
    def load_labels(file_name):
        with gzip.open(os.path.join(data_dir, file_name),'rb') as f:
            return np.frombuffer(f.read(),np.uint8, offset=8)
        
    x_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    x_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz" )

    indices = np.arange(x_train.shape[0])
    ##随机
    np.random.shuffle(indices)
    
    n_train = int(x_train.shape[0]*(1 - val_split))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    return x_train, y_train, x_val, y_val,x_test, y_test

def one_hot_process(x,y):
    x=x.astype(np.float32)/ 255.0
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size),y] = 1
    x=x.T
    y_one_hot = y_one_hot.T
    return x,y_one_hot

if __name__=="__main__":
    data_dir ='./fashion-mnist'
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir, training=True, val_split=0.1)
    print("Data loaded successfully!")
    x_train, y_train = one_hot_process(x_train, y_train)
    x_val, y_yal = one_hot_process(x_val,y_val)
    x_test, y_test = one_hot_process(x_val,y_val)