import torch, os
import aiter, tk_kernel_bkwd, tk_kernel_bkwd_prep
torch.manual_seed(0)
B=int(os.environ.get("BB","16")); H=int(os.environ.get("HH","64")); H_KV=int(os.environ.get("HKV","8"))
N=int(os.environ.get("NN","4096")); D=int(os.environ.get("DD","64")); causal=int(os.environ.get("CAUSAL","1"))
dev,dt='cuda',torch.bfloat16
def gen(s):
    t=torch.randn(s,dtype=dt,device=dev); m=torch.norm(t,dim=-1,keepdim=True)
    return (t*(torch.randn(m.shape,dtype=dt,device=dev)*0.1+10)/m).contiguous()
Q=gen((B,H,N,D));K=gen((B,H_KV,N,D));V=gen((B,H_KV,N,D));dO=gen((B,H,N,D))
Qa=Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka=K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va=V.transpose(1,2).contiguous().detach().requires_grad_(True)
out,lse=aiter.flash_attn_func(Qa,Ka,Va,causal=bool(causal),return_lse=True,deterministic=False)
Q_tk=Q.transpose(1,2).bfloat16().contiguous();K_tk=K.transpose(1,2).bfloat16().contiguous()
V_tk=V.transpose(1,2).bfloat16().contiguous();dO_tk=dO.transpose(1,2).bfloat16().contiguous()
O_tk=out.detach().bfloat16().contiguous()
L_tk=lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()
def hk_bwd():
    dQi=torch.zeros((B,N,H,D),device=dev,dtype=dt);dQ=torch.zeros((B,N,H,D),device=dev,dtype=dt)
    dK=torch.zeros((B,N,H_KV,D),device=dev,dtype=dt);dV=torch.zeros((B,N,H_KV,D),device=dev,dtype=dt)
    delta=torch.zeros((B,H,N,1),device=dev).float().transpose(-1,-2).contiguous()
    tk_kernel_bkwd_prep.dispatch_prep(O_tk,dO_tk,delta)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQi,dK,dV,L_tk,delta)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQi,dQ)
for _ in range(2): hk_bwd()   # warmup
torch.cuda.synchronize()
for _ in range(3): hk_bwd()   # traced region
torch.cuda.synchronize()
