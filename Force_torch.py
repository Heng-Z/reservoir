# %%
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
def sprandn(N1,N2,p):
    # import sparse 
    import scipy.sparse as sparse
    # import stats
    import scipy.stats as stats
    rvs = stats.norm(loc=0, scale=1).rvs
    S = sparse.random(N1, N2, density=p, data_rvs=rvs)
    return S.toarray()

def main(lr = 1e-3):
    N = 1000
    Nout = 1
    pgg = 0.1
    pz = 1
    wash = 0.5
    nt = 2
    alpha = 1e-3
    fb = 1
    g = 1.2
    learning_rate = lr
    print(type(lr))
    dtype = torch.float
    device = torch.device("cuda")
    # initiate connection
    M = sprandn(N,N,pgg)
    M = torch.tensor(M/max(abs(np.linalg.eig(M)[0])*g),dtype=dtype,device=device,requires_grad=False)
    wf = fb*(2*torch.rand(N,1,device=device)-1)
    wo = torch.randn(N,1,device=device,requires_grad=True)
    # print('leaf?',wo.is_leaf)
    b = torch.rand(N,1,device=device,requires_grad=False)
    #prepare data
    data = scipy.io.loadmat('/home/heng/software/easyesn/mytest/lorenz_data.mat')['xdat']
    data = (data - np.mean(data,axis = 0))/np.std(data,axis=0)
    ft = torch.tensor(data[0:5000,0],dtype=dtype,device=device)
    fts = torch.tensor(data[5000:6500,0],dtype=dtype,device=device)
    # print(ft.shape)
    L = 5000
    L_test = 1500
    #initiate state
    x = torch.randn(N,1,device=device)
    y = torch.randn(N,1,device=device)
    y_rec = torch.zeros(ft.shape,device=device)
    y_test_rec = torch.zeros((Nout,L_test),device=device)
    for i in range(10):
        # print(i)
        x = x*(1-wash) + wash * torch.tanh(torch.matmul(M,x)+b)
        y = torch.matmul(wo.T,x)
        if np.mod(i,nt) == 0:
            Loss = (y - ft[i])**2 + alpha *torch.matmul(w
            o.T,wo)
            # print(Loss)
            Loss.backward()
            # print('leaf?',wo.is_leaf)
        # with torch.no_grad():
            with torch.no_grad():
                print(wo.grad)
                wo -= learning_rate*wo.grad 
                wo.grad = None
        y_rec[i] = y

    #testing
    # with torch.no_grad():
    for i in range(L_test):
        x = x*(1-wash) + wash * torch.tanh(torch.matmul(M,x)+b)
        y = torch.matmul(wo.T,x)
        y_test_rec[0,i] = y
    # plt.subplot(2,1,1)
    # print(y_test_rec)
    # y_test_rec_np =np.zeros((Nout,L_test))
    # y_test_rec_np[...] = y_test_rec.cpu().detach().numpy()
    print(y_test_rec)
    plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, lr={}".format(g,fb,pz,pgg,alpha,learning_rate), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    # plt.plot(y_test_rec.cpu().detach().numpy().T,'r',fts.cpu().detach().numpy(),'b')
    plt.plot(y_test_rec.cpu().detach().numpy().T,'r')
    filename = './image/torch/lorenz1_' + ct +'.jpeg'
    plt.savefig(filename,dpi=300)
    
    # return y_test_rec
if __name__ == '__main__':
    import datetime
    for lr in [1.0,0.5,0.1,0.05,0.01,0.005,0.001]:
        main(lr)
    
    




    
        






