import random
import torch
import numpy as np
import torch.nn.functional as F

def simple_hard_neg(samples_p, mean_p, samples_n, mean_n, beta=0.5, sample_size=500):
    """
    簡化版 hard_neg，使用歐氏距離計算困難負樣本
    INPUT:
    samples_p = 正樣本的張量，形狀為 (num_samples_p, embedding_dimension)
    mean_p = 正樣本的均值向量
    samples_n = 負樣本的張量，形狀為 (num_samples_n, embedding_dimension)
    mean_n = 負樣本的均值向量
    beta = 用於負樣本的凸組合權重
    sample_size = 每次生成的樣本數量

    OUTPUT:
    (hn_mu_n, hn_sig_n) = (均值, 協方差矩陣) 的張量
    """
    num_pos, num_neg = samples_p.size(0), samples_n.size(0)
    with torch.no_grad():
        # 將張量轉換為 NumPy 陣列
        t_p, mu_p, t_n, mu_n = samples_p.numpy(), mean_p.numpy(), samples_n.numpy(), mean_n.numpy()

        # 計算負樣本到正樣本均值的歐氏距離，並選擇距離最遠的樣本
        dists_from_p = np.linalg.norm(t_n - mu_p, axis=1)
        farthest_neg_idx = np.argmax(dists_from_p)
        farthest_neg = t_n[farthest_neg_idx]

        # 生成新的負樣本
        new_negs = np.random.uniform(mu_n, farthest_neg, size=(3 * num_neg, mu_n.shape[0]))
        
        if new_negs.size == 0:
#             print("No negative samples generated. Please check the input parameters and conditions.")
            return None, None

        # 轉換回 PyTorch 張量
        all_negs = torch.from_numpy(new_negs).float()
        all_negs = F.normalize(all_negs, p=2, dim=1)
        hn_mu_n = torch.mean(all_negs, dim=0)
        hn_sig_n = torch.mm((all_negs - hn_mu_n).t(), (all_negs - hn_mu_n)) / all_negs.size(0)

    return hn_mu_n, hn_sig_n

# 主要在篩選「困難負樣本」，那些位於特定區域內的困難負樣本。這些樣本在特徵空間中與正樣本接近，但仍然屬於負樣本。
def hard_neg(samples_p, mean_p, cov_p, samples_n, mean_n, cov_n, beta=0.5):
    """
    INPUT:
    t_p = tensor of shape num_examples_of_positive_class*embedding_dimension
    mu_p = tensor denoting mean of positive class
    sig_p = tensor denoting cov matrix of positive class
    t_n = tensor of shape num_examples_of_negative_class*embedding_dimension 
    mu_n = tensor denoting mean of negative class
    sig_n = tensor denoting cov matrix of negative class
    beta = int random weight for convex combination sampled from Uniform (0, beta)

    OUTPUT:
    (hn_mu_n, hn_sig_n)=(mean, cov)[tensors] of distributiion incoporating hard negative mining
    """
    num_pos, num_neg = samples_p.size()[0], samples_n.size()[0]
    with torch.no_grad():
        # NumPy 轉換
        t_p, mu_p, sig_p, t_n, mu_n, sig_n = samples_p.numpy(), mean_p.numpy(), cov_p.numpy(), samples_n.numpy(), mean_n.numpy(), cov_n.numpy()
        sig_p_inv, sig_n_inv = np.linalg.pinv(sig_p), np.linalg.pinv(sig_n)
        sig_p_inv, sig_n_inv = sig_p_inv/1e10, sig_n_inv/1e10
        
        # 計算樣本到對應均值的馬氏距離並找到錨點 Mahalanobis Distance
        dists_from_p = np.diag(np.matmul((t_n-mu_p),np.matmul(sig_p_inv,(t_n-mu_p).T)))
        anchor_n = t_n[np.argmin(dists_from_p),:]
        dists_from_n = np.diag(np.matmul((t_p-mu_n),np.matmul(sig_n_inv,(t_p-mu_n).T)))
        anchor_p = t_p[np.argmin(dists_from_n),:]
        
        # anchor_n: 最接近正樣本分佈中心 mu_p 的負樣本。
        # anchor_p: 最接近負樣本分佈中心 mu_n 的正樣本。
        # define normal vector
        normal_vec=anchor_n-anchor_p
        # define constraints
        l, u = np.dot(normal_vec, anchor_p), np.dot(normal_vec, anchor_n)

        # sample till you get 3*num_neg negatives satisfying constraints
        negs=np.ndarray(0)
        while (True) :
            sample = np.random.multivariate_normal(mu_n, sig_n, 1000) # 從常態分佈中取出樣本 1000 個
            checker = np.dot(sample, normal_vec.T) # 內積
            
            # 篩選符合條件的樣本
            sample = sample[np.logical_and(checker<=u,checker>=l)]
            if negs.shape[0]==0:
                negs = sample
            else :
                negs = np.vstack((negs, sample))
            # break out of loop when you get enough samples
            if negs.shape[0]>=3*num_neg:
                break
        
        # sort samples acc to mahalanobis dist to neg distribution
        dists = np.diag(np.matmul((negs-mu_n),np.matmul(sig_n_inv,(negs-mu_n).T)))
        negs = negs[np.argsort(dists)] #sorts sampled points in ascedning order of mahalanobis dist from negative distribution

        # extract top num_neg samples for checking distance constraint to obtain qualified negs
        sampled_negs = negs[0:num_neg]
        dist_n_achor = np.matmul((anchor_n-mu_p),np.matmul(sig_p_inv,(anchor_n-mu_p).T))
        dist_of_sampled_from_p = np.diag(np.matmul((sampled_negs-mu_p),np.matmul(sig_p_inv,(sampled_negs-mu_p).T)))
        disqualified_indices=dist_of_sampled_from_p>dist_n_achor
        disqualified_negs = sampled_negs[disqualified_indices]
        inter_negs=sampled_negs#intialising
        for i in range(sum(disqualified_indices)):
            w= np.random.uniform(beta-0.2,beta+0.2)
            temp=random.choices(inter_negs, k=2)
            new_neg = w*temp[0]+(1-w)*temp[1]
            inter_negs=np.vstack((new_neg, inter_negs))
            
    # 將樣本轉換回 PyTorch
    all_negs=torch.from_numpy(inter_negs)
    # NORMALIZE EMBEDDINGS WITH L2 NORM=1
    all_negs = F.normalize(all_negs, p=2, dim=1)
    hn_mu_n = torch.mean(all_negs, dim=0)
    hn_sig_n = torch.mm((all_negs-hn_mu_n).t(), ((all_negs-hn_mu_n)))/(all_negs.size()[0])

    return hn_mu_n.float(), hn_sig_n.float()
