import torch 
import numpy as np
from utils.patchCL.hard_neg_mining import simple_hard_neg, hard_neg

def simple_PCGJCL(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
    """
    INPUT:
    patch_list = list of length num_class_in_patches containing tensors t_i
    t_i = tensor of shape num_samples_class_i * dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT:
    loss = tensor
    """
    N = len(patch_list)
    total_samples = 0
    mean_list = []
    cov_list = []
    loss = torch.tensor([0.0], dtype=torch.float32)
    num_classes_in_batch = 0

    # 計算每個類別的均值和協方差矩陣
    for i in range(N):
        if patch_list[i] is not None:
            t_i = torch.stack(embd_queues[i], dim=0)
            total_samples += patch_list[i].size(0)
            num_classes_in_batch += 1
            mu = torch.mean(t_i, dim=0)
            mean_list.append(mu)
            sig = torch.mm((t_i - mu).t(), (t_i - mu)) / t_i.size(0)
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    g_count = 0
    den_lists = [[] for _ in range(N)]
    pos_lists = [None for _ in range(N)]
    
    for i in range(N):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i = patch_list[i], mean_list[i], cov_list[i]
            num = (-torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0)) / tau
            l_count = 0
            # 遍歷所有其他類別，計算負樣本對比損失
            for j in range(N):
                if patch_list[j] is not None and i != j:
                    t_j, mu_j, sig_j = torch.stack(embd_queues[j], dim=0), mean_list[j], cov_list[j]
                    # 使用簡化的 hard neg 函數
                    hn_mu_j, hn_sig_j = simple_hard_neg(t_i, mu_i, t_j, mu_j, 0.5, sample_size=500)
                    
                    if hn_mu_j is None or hn_sig_j is None:
                        print(f"Skipping class pair ({i}, {j}) due to lack of negative samples.")
                        continue

                    den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1))) / tau) + (0.5 * lamb / (tau ** 2)) * (torch.diag(torch.mm(t_i, torch.mm(hn_sig_j, t_i.t()))).view(-1, 1))
                    den_lists[i].append(den_neg)
                elif patch_list[j] is not None:
                    den_pos = torch.mm(t_i, mu_i.view(emb_dim, 1)) / tau + (0.5 * lamb / (tau ** 2)) * (torch.diag(torch.mm(t_i, torch.mm(sig_i, t_i.t()))).view(-1, 1))
                    pos_lists[i] = den_pos
        
            a = pos_lists[i]
            res = torch.zeros(a.size())
            res += torch.exp(a)
            for d in den_lists[i]:
                res += (psi / num_classes_in_batch) * torch.exp(d)
            den = torch.sum(torch.log(res), dim=0)
            if g_count == 0:
                loss = num + den
                g_count += 1
            else:
                loss += num + den

    loss = loss / total_samples
    return loss

def PCGJCL(patch_list, embd_queues, emb_dim, tau, lamb, psi=4096):
  
    """
    INPUT :
    patch_list = list of length num_class_in_patches containing tensors t_i
    t_i = tensor of shape num_samples_class_i*dim
    tau = temperature parameter
    lamb = lambda : scaling parameter
    alpha = negative weighting term 

    OUTPUT :
    loss = tensor
    """
    N= len(patch_list)
    total_samples=0
    
    mean_list=[]# 每一個樣本的平均值
    cov_list=[] # 每一個樣本的標準差
    
    loss = torch.tensor([0])
    num_classes_in_batch=0
    
    # 2 類別 N = 2
    for i in range(N):
        if patch_list[i] is not None:
            # 把類別中的 tensor 集合再一起，background 有 4096 個 tensor，變成一個 [tensor_0, tensor_1, .....]
            t_i = torch.stack(embd_queues[i], dim=0)
            
            # 類別樣本數
            total_samples += patch_list[i].size()[0]
            num_classes_in_batch+=1
            
            # 機算該類別的 embedding 平均
            mu=torch.mean(t_i, dim=0)
            mean_list.append(mu)
            
            # 計算該類別的樣本嵌入協方差矩陣 sig。這裡的計算方法是先減去均值，然後計算樣本之間的內積，再除以樣本數量。
            sig =torch.mm((t_i-mu).t(), (t_i-mu))/(t_i.size()[0])
            cov_list.append(sig)
        else:
            mean_list.append(None)
            cov_list.append(None)

    g_count=0
    
    den_lists=[] # 副樣本損失
    pos_lists=[] # 正本損失
    for i in range(N):
        if patch_list[i] is not None:
            t_i, mu_i, sig_i=patch_list[i], mean_list[i], cov_list[i] # 取得該類別的樣本嵌入、均值和協方差矩陣。
            
            # 對比學習，該類別的正樣本。torch.mm(t_i, mu_i.view(emb_dim, 1)) 計算內積，計算每一個向量與平均的距離，並使用 torch.sun 求和
            num = (-torch.sum(torch.mm(t_i, mu_i.view(emb_dim, 1)), dim=0))/tau
            l_count=0 
            
            #Iterate over neg classes for a particluar class
            # 計算副樣本，i!=j，只要不是當前類別的。
            for j in range(N):
                if patch_list[j] is not None:
                    if i!=j:
                        t_j, mu_j, sig_j= torch.stack(embd_queues[j], dim=0), mean_list[j], cov_list[j]
                        #get hard neg mu_j,sig_j
                        hn_mu_j, hn_sig_j = hard_neg(t_i, mu_i, sig_i, t_j, mu_j, sig_j, 0.5)# GET HARD NEGS HERE
                        #hn_mu_j, hn_sig_j = mu_j, sig_j # USE THIS FOR NO HARD NEGS
                        if l_count==0:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            l_count+=1
                            den_lists.append([den_neg])
                        else:
                            den_neg = ((torch.mm(t_i, hn_mu_j.view(emb_dim, 1)))/tau) + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(hn_sig_j, t_i.t())))).view(-1,1))
                            den_lists[i].append(den_neg)
                
                    else:
                        den_pos = torch.mm(t_i, mu_i.view(emb_dim, 1))/tau + (0.5*lamb/(tau**2))*(torch.diag((torch.mm(t_i, torch.mm(sig_i, t_i.t())))).view(-1,1))
                        pos_lists.append(den_pos)
                # else:
                #     den_lists[i].append(None)
        
            a=pos_lists[i]
            res=torch.zeros(a.size())
            res+=torch.exp(a)       
            for d in den_lists[i]:
              res+=(psi/num_classes_in_batch)*torch.exp(d)# ADDED WEIGHT FOR NEG TERMS HERE ....psi*den_neg+den_pos
            den = torch.sum(torch.log(res), dim=0)
            if g_count==0:
                loss= num+den
                g_count+=1
            else:
                loss+=num+den
        
        else:
          pos_lists.append(None)
          den_lists.append(None)

    loss = loss/total_samples
    
    return loss
