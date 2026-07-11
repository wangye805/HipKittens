import torch, os
import aiter, tk_kernel_bkwd, tk_kernel_bkwd_prep
torch.manual_seed(0)
N=256; B,H,H_KV,D=1,8,1,64
dev,dt='cuda',torch.bfloat16
def gen(s):
    t=torch.randn(s,dtype=dt,device=dev); m=torch.norm(t,dim=-1,keepdim=True)
    return (t*(torch.randn(m.shape,dtype=dt,device=dev)*0.1+10)/m).contiguous()
Q=gen((B,H,N,D)); K=gen((B,H_KV,N,D)); V=gen((B,H_KV,N,D)); dO=gen((B,H,N,D))
Qa=Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka=K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va=V.transpose(1,2).contiguous().detach().requires_grad_(True)
out,lse=aiter.flash_attn_func(Qa,Ka,Va,causal=True,return_lse=True,deterministic=False)
out.backward(dO.transpose(1,2).contiguous()); dQr=Qa.grad  # (B,N,H,D)
# run kernel
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
# reference dS decomposition, head 0
h=0; scale=1.0/(D**0.5)
Qf=Q[0,h].float(); Kf=K[0,0].float(); Vf=V[0,0].float(); dOf=dO[0,h].float()
S=(Qf@Kf.t())*scale
mask=torch.tril(torch.ones(N,N,device=dev)).bool()
S=S.masked_fill(~mask,float('-inf'))
P=torch.softmax(S,dim=-1); Of=P@Vf
dP=dOf@Vf.t(); dlt=(dOf*Of).sum(-1,keepdim=True)
dS=P*(dP-dlt)
dQrecon=(dS@Kf)*scale  # (N,D)
for q in [128, 200, 255]:
    tot=dQrecon[q,0].item(); tk=dQ_tk[0,q,h,0].item(); af=dQr[0,q,h,0].item()
    blk=[ (dS[q,64*b:64*b+64]*Kf[64*b:64*b+64,0]).sum().item()*scale for b in range(4)]
    print(f"q={q}: aiter={af:+.5f} recon={tot:+.5f} tk={tk:+.5f} | blocks(KV) col0: "
          + " ".join(f"b{b}={blk[b]:+.5f}" for b in range(4)))
