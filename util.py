import autograd.numpy as np
from autograd.util import flatten_func
from autograd import grad as compute_grad  
import matplotlib.pyplot as plt

def gradient_descent(g,w,alpha,max_its,beta,version):
    g_flat, unflatten, w = flatten_func(g, w)
    grad = compute_grad(g_flat)
    w_hist = []
    w_hist.append(unflatten(w))
    z = np.zeros((np.shape(w)))     
    
    for k in range(max_its):   
        grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
            grad_eval /= grad_norm
            
        z = beta*z + grad_eval
        w = w - alpha*z
        w_hist.append(unflatten(w))

    return w_hist

def find_softmax_costs(weight_history,g):
    cost_arr=[]
    for w in weight_history:
        cost_arr.append(g(w))
    return cost_arr

def find_all_counts(weight_history):
    count_hist = []
    for p in range(0,len(weight_history)):
        count_hist.append(find_count(weight_history[p]))
    return count_hist

def plot_two(counts,cost):
    ax1 = plt.subplot(1,2,1)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('misclasifications')
    plt.plot(counts)
    ax2 =plt.subplot(1,2,2)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('cost')
    plt.plot(cost)
    plt.subplots_adjust(wspace=.5)
    plt.show()