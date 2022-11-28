from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    
    ## X:(500,3073)
    ## W:(3073,10)
    ## S = X*W (500,10)
    

    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    ## 初始化训练集的规模
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    S = X.dot(W)
    
    #############################################################################
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！                                                           
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ## loss-function :
    ## Li = -log(e^Syi/sum(e^Sj)) 其中yi为第i个样本的真实的标签

    ## Loss:
    for i in range(num_train):
        s_yi = np.exp(S[i][y[i]]) ## 分子
        sum_i = sum(np.exp(S[i])) ## 分母
        res = np.log(s_yi*1.0 / sum_i)
        loss -= res
    
    ## 正则化损失
    loss /= num_train
    loss += reg * np.sum(np.square(W))
    
    ## dW:
    ## dL/dW = dL/dS*dS/dW
    """
    Li = -log(Pk)
    Pk = e^Sk/sum(e^Sj) 
    若 k==yi
    则 dLi/dSk = Pk-1
    若 k!=yi
    则 dLi/dSk = Pk
    
    dS/dW = X.T
    dLi/dW = Pk*X[i].T||(Pk-1)*X[i].T
    """

    for i in range(num_train):
        sum_i = sum(np.exp(S[i])) ## 分母
        for j in range(num_classes):
            s_i = (np.exp(S))[i][j]
            res = (s_i*1.0 / sum_i)
            dW[:,j] += res*X[i]
            if j == y[i]:
                dW[:,j] -= X[i]
                
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: 不使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ## 初始化训练集的规模
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    S = X.dot(W)
        
    ## Loss:
    
    correct = S[np.arange(num_train), y] ##取出正确分类的值并组为一列
    res = np.exp(correct) / np.sum(np.exp(S),axis = 1)
   
    loss -= np.sum(np.log(res))


    ## 正则化损失
    loss /= num_train
    loss += reg * np.sum(np.square(W))
    
    ## dW:
    
    res1 = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)
    res2 = np.zeros_like(S)
    res2[np.arange(num_train), y] = 1 
    res2 = X.T.dot(res2)
    dW += X.T.dot(res1) ## 非正确分类项
    dW -= res2 ## 处理正确分类项




    dW /= num_train
    dW += 2*reg*W
    

    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
