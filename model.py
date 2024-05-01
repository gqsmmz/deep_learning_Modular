import numpy as np

class DeepNet:
    def __init__(self,layer_dims,activation_type='relu') :
        self.parameters=self.initialize_parameters(layer_dims)
        self.activation_type=activation_type

    def initialize_parameters(self,layer_dims):
        parameters={}
        length=len(layer_dims)
        for i in range(1,length):
            W_name='W'+str(i)
            b_name='b'+str(i)
            input_size=layer_dims[i]
            hidden_size=layer_dims[i-1]
            parameters[W_name]=np.random.randn(input_size,hidden_size)*0.01
            parameters[b_name]=np.zeros((input_size,1))
        return parameters
    
    def activate(self,z):
        if self.activation_type=='relu':
            return relu(z)
        elif self.activation_type=='sigmoid':
            return sigmoid(z)
        elif self.activation_type=='tanh':
            return tanh(z)
        elif self.activation_type=='leaky_relu':
            return leaky_relu(z)
        else:
            raise ValueError("Unnamed activation function")
    
    def d_activation(self,z):
        if self.activation_type=='relu':
            return relu_derivative(z)
        elif self.activation_type=='sigmoid':
            return sigmoid_derivative(z)
        elif self.activation_type=='tanh':
            return tanh_derivative(z)
        elif self.activation_type=='leaky_relu':
            return leaky_relu_derivative(z)
        elif self.activation_type=='softplus':
            return softplus_derivative(z)
        else:
            raise ValueError("unsupported activation function")
        
    def forward(self,x):
        ##初始化
        caches=[]
        act=x
        L=len(self.parameters)//2
        ##一层层往前传
        a_prey=act
        z_out=np.dot(self.parameters['W1'],a_prey)+self.parameters['b1']
        act=self.activate(z_out)
        caches.append((a_prey,z_out,act))
        z_out=np.dot(self.parameters['W2'],act)+self.parameters['b2']
        act=softmax(z_out)
        caches.append((act,z_out,None))
        return act,caches
    
    def backward(self,x,y,caches,l2_reg,batch_size):
        ##初始化
        grads={}
        m=y.shape[1]
        ##反向一层层求导
        ##softmax和倒数第一层fc
        L=len(caches)
        aL,zL,_=caches[-1]
        dz=aL-y
        grads["dW2"]=np.dot(dz,caches[0][2].T)/m
        grads["db2"]=np.sum(dz,axis=1,keepdims=True)/m
        ##activation和倒数第二层fc
        a_prev,z,a=caches[0]
        da=np.dot(self.parameters["W2"].T,dz)
        dz=da*self.d_activation(z)
        grads["dW1"]=np.dot(dz,a_prev.T)/m
        grads["db1"]=np.sum(dz,axis=1,keepdims=True)/m

        for key in self.parameters.keys():
            grads["d"+key] += (l2_reg / batch_size) * self.parameters[key]
        return grads
    
    def update_params(self,grads,learning_rate):
        deep_len=len(self.parameters)//2
        for i in range(1,deep_len+1):
            self.parameters["W"+str(i)] -= learning_rate*grads["dW"+str(i)]
            self.parameters["b"+str(i)] -= learning_rate*grads["db"+str(i)]
    
    def load_parameters(self, file_path):
        loaded_params = np.load(file_path)
        for key in self.parameters:
            if key in loaded_params:
                self.parameters[key] = loaded_params[key]
            

## 激活函数
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha*z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)


def softplus_derivative(z):
    return sigmoid(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)