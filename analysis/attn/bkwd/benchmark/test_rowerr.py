import torch, os
import aiter, tk_kernel_bkwd, tk_kernel_bkwd_prep
torch.manual_seed(0)
N=int(os.environ.get("NN","256")); B,H,H_KV,D=1,8,1,64
dev,dt='cuda',torch.bfloat16
def gen(s):
    t=torch.randn(s,dtype=dt,device=dev); m=torch.norm(t,dim=-1,keepdim=True)
    return (t*(torch.randn(m.shape,dtype=dt,device=dev)*0.1+10)/m).contiguous()
Q=gen((B,H,N,D)); K=gen((B,H_KV,N,D)); V=gen((B,H_KV,N,D)); dO=gen((B,H,N,D))
Qa=Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka=K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va=V.transpose(1,2).contiguous().detach().requires_grad_(True)
out,lse=aiter.flash_attn_func(Qa,Ka,Va,causal=True,return_lse=True,deterministic=False)
out.backward(dO.transpose(1,2).contiguous()); dQr=Qa.grad
Q_tk=Q.transpose(1,2).bfloat16().contiguous();K_tk=K.transpose(1,2).bfloat16().contiguous()
V_tk=V.transpose(1,2).bfloat16().contiguous();dO_tk=dO.transpose(1,2).bfloat16().contiguous()
O_tk=out.detach().bfloat16().contiguous()
L_tk=lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()
delta=(dO_tk.float()*O_tk.float()).sum(-1); delta_tk=delta.permute(0,2,1).unsqueeze(2).contiguous()
dQ_in=torch.zeros_like(Qa.grad).bfloat16().transpose(1,2).contiguous()
dK_tk=torch.zeros_like(Ka.grad).bfloat16().contiguous();dV_tk=torch.zeros_like(Va.grad).bfloat16().contiguous()
tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQ_in,dK_tk,dV_tk,L_tk,delta_tk); torch.cuda.synchronize()
dQ_tk=torch.zeros_like(dQr).bfloat16().contiguous()
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in,dQ_tk); torch.cuda.synchronize()
tk=dQ_tk[0,:,0,:].float(); rf=dQr[0,:,0,:].float()
# per-row relative error at col 0 (bad) and col 2 (good), averaged over heads
def relerr(col):
    a=dQ_tk[0,:,:,col].float(); b=dQr[0,:,:,col].float()
    return ((a-b).abs()/(b.abs()+1e-6))
e0=relerr(0).mean(dim=1); e2=relerr(2).mean(dim=1)  # per-row over heads
for q in [0,1,2,3,4,8,16,32,64,128,200,255]:
    print(f"row {q:3d}: col0 relerr={e0[q].item():.4f}  col2 relerr={e2[q].item():.4f}")

print("=== rows with col0 relerr > 0.01 ===")
bad = (e0 > 0.01).nonzero().flatten().tolist()
print("bad rows:", bad)
print("count:", len(bad), "of", N)
