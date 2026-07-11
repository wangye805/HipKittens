import torch, os
import aiter, tk_kernel_bkwd, tk_kernel_bkwd_prep
torch.manual_seed(0)
N=int(os.environ.get("NN","256")); B,H,H_KV,D=1,8,1,64
dev,dt='cuda',torch.bfloat16
def gen(s):
    t=torch.randn(s,dtype=dt,device=dev); m=torch.norm(t,dim=-1,keepdim=True)
    return (t*(torch.randn(m.shape,dtype=dt,device=dev)*0.1+10)/m).contiguous()
Q=gen((B,H,N,D)); K=torch.ones((B,H_KV,N,D),dtype=dt,device=dev); V=gen((B,H_KV,N,D)); dO=gen((B,H,N,D))
Qa=Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka=K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va=V.transpose(1,2).contiguous().detach().requires_grad_(True)
out,lse=aiter.flash_attn_func(Qa,Ka,Va,causal=True,return_lse=True,deterministic=False)
out.backward(dO.transpose(1,2).contiguous())
dQr=Qa.grad
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
cos=torch.nn.functional.cosine_similarity(dQr.flatten().float(),dQ_tk.flatten().float(),dim=0).item()
# D-variance: with K=ones, each dQ[b,n,h,:] should be constant across D
var_tk=dQ_tk.float().var(dim=-1).mean().item()
var_ref=dQr.float().var(dim=-1).mean().item()
print(f"K=ones: dQ cos={cos:.6f}  dQ_tk D-var(mean)={var_tk:.3e}  dQ_ref D-var(mean)={var_ref:.3e}")
print("dQ_tk[0,10,0,:8]:", dQ_tk[0,10,0,:8].float().tolist())
print("dQ_rf[0,10,0,:8]:", dQr[0,10,0,:8].float().tolist())
