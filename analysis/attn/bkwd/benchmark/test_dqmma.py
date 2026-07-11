import torch, dq_mma_test
dev='cuda'; N,Ds,Q=256,64,16
torch.manual_seed(0)
K=torch.randn(1,N,1,Ds,dtype=torch.bfloat16,device=dev)
dP=torch.randn(1,N,1,Q,dtype=torch.bfloat16,device=dev)
dQo=torch.zeros(1,64,1,16,dtype=torch.float32,device=dev)
dq_mma_test.dispatch(K,dP,dQo); torch.cuda.synchronize()
print("dQo abs-mean:", dQo.float().abs().mean().item())
Kf=K[0,:,0,:].float(); dPf=dP[0,:,0,:].float()
for w in range(4):
    # dQ_i_T[dl][q] stored at dQo[0, 16w+dl, 0, q]
    tkT=dQo[0,16*w:16*w+16,0,:].float()   # (Dslice, Q)
    refT=(dPf.t() @ Kf[:,16*w:16*w+16]).t()  # (Dslice,Q)
    c=torch.nn.functional.cosine_similarity(tkT.flatten(),refT.flatten(),dim=0).item()
    print(f"warp {w}: cos={c:.6f}")
