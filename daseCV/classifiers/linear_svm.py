from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) ## 乘上W计算处X_i分类到不同类别下的分数
        correct_class_score = scores[y[i]] ## 取出正确类别下的分数
        for j in range(num_classes): ## 历遍类别求和得到损失值
            if j == y[i]:
                continue ## 正确的类别下的值不予计算
            margin = scores[j] - correct_class_score + 1 # note delta = 1 ## (WX_i - WX_yi + 1)
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] # dW计算
                dW[:,y[i]] += -X[i] # dW计算 此处i = y[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5*reg * np.sum(W * W)
    

    #############################################################################
    # TODO：
    # 计算损失函数的梯度并将其存储为dW。
    # 与其先计算损失再计算梯度，还不如在计算损失的同时计算梯度更简单。
    # 因此，您可能需要修改上面的一些代码来计算梯度。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #     正则化梯度项
    dW /= num_train
    dW += reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW
 


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO: 
    # 实现一个向量化SVM损失计算方法,并将结果存储到loss中
    #############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 先计算损失
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    scores = X.dot(W)
#     print(scores.shape)
    
    ## scores为500*10的矩阵，代表500个样本对应为一个分类下的分数
    
    ##取出500个样本对应的正确的分类的分数
    scores_correct = scores[np.arange(num_train),y]
    ## [[],
    ##  [],
    ##  [].
    ##  [],
    ##  ...
    ##  []]
    scores_correct = np.reshape(scores_correct,(num_train, -1))
    loss_martix = scores - scores_correct + 1.0 ## 计算loss
    loss_martix[np.arange(num_train),y] = 0 ## 排除正确分类处的计算
    loss_martix[loss_martix < 0] = 0          ## 实现max()运算
#     loss_martix = np.sum(loss_martix,axis = 1)   ## 行求和得到500个样本的loss 此处的loss还是500*1的矩阵
    loss += np.sum(loss_martix)
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5*reg * np.sum(W * W)
    
    
    
#     for i in range(num_train):
#         scores = X[i].dot(W) ## 乘上W计算处X_i分类到不同类别下的分数
#         correct_class_score = scores[y[i]] ## 取出正确类别下的分数
#         for j in range(num_classes): ## 历遍类别求和得到损失值
#             if j == y[i]:
#                 continue ## 正确的类别下的值不予计算
#             margin = scores[j] - correct_class_score + 1 # note delta = 1 ## (WX_i - WX_yi + 1)
#             if margin > 0:
#                 loss += margin
#                 dW[:,j] += X[i] # dW计算
#                 dW[:,y[i]] += -X[i] # dW计算 此处i = y[i]

    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                              
    # 实现一个向量化的梯度计算方法,并将结果存储到dW中                                
    #                                                                           
    # 提示:与其从头计算梯度,不如利用一些计算loss时的中间变量                                    
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ## dW[i,j] = sum(yk!=j){Xki}-sum(k==yj){Xki}
    
    loss_martix[loss_martix > 0] = 1.0
    
    ## 损失值大于0处代表此处的运算为WXi-WXyi+1 此处存在W的梯度
    
    row_sum = np.sum(loss_martix, axis=1)
    
    ## 对于正确分类处W的梯度是-Xyi,
    loss_martix[np.arange(num_train), y] = -row_sum
    
    dW = X.T.dot(loss_martix)
    
    dW /= num_train
    dW += reg*W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
