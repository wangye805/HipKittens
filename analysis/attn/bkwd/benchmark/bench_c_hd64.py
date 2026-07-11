import torch, os, math, time
import aiter
import tk_kernel_bkwd
import tk_kernel_bkwd_prep

torch.manual_seed(0)
B   = int(os.environ.get("BB",  "16"))
H   = int(os.environ.get("HH",  "64"))
H_KV= int(os.environ.get("HKV", "8"))
N   = int(os.environ.get("NN",  "4096"))
D   = int(os.environ.get("DD",  "64"))
causal = int(os.environ.get("CAUSAL", "1"))
WARM = int(os.environ.get("WARM", "20"))
ITERS= int(os.environ.get("ITERS","50"))
dev, dt = 'cuda', torch.bfloat16

def flops(mode="bwd"):
    f = 4 * B * N**2 * H * D // (2 if causal else 1)
    return 2.5 * f  # bwd
def eff(fl, ms): return (fl/1e12)/(ms/1e3)

def gen(shape):
    t = torch.randn(shape, dtype=dt, device=dev)
    mag = torch.norm(t, dim=-1, keepdim=True)
    return (t*(torch.randn(mag.shape, dtype=dt, device=dev)*0.1+10)/mag).contiguous()

Q = gen((B,H,N,D)); K = gen((B,H_KV,N,D)); V = gen((B,H_KV,N,D)); dO = gen((B,H,N,D))
Qa = Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka = K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va = V.transpose(1,2).contiguous().detach().requires_grad_(True)
dOa = dO.transpose(1,2).contiguous()
out, lse = aiter.flash_attn_func(Qa, Ka, Va, causal=bool(causal), return_lse=True, deterministic=False)

s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)

# ---- aiter bwd timing ----
for _ in range(WARM):
    o2,_ = aiter.flash_attn_func(Qa, Ka, Va, causal=bool(causal), return_lse=True, deterministic=False)
    o2.backward(dOa, retain_graph=True)
t_aiter=[]
for _ in range(ITERS):
    o2,_ = aiter.flash_attn_func(Qa, Ka, Va, causal=bool(causal), return_lse=True, deterministic=False)
    torch.cuda.synchronize(); s.record()
    o2.backward(dOa, retain_graph=True)
    e.record(); torch.cuda.synchronize(); t_aiter.append(s.elapsed_time(e))
ms_aiter=sum(t_aiter)/len(t_aiter)

# ---- HK bwd timing (prep + bwd_combined + dq_shuffle) ----
Q_tk = Q.transpose(1,2).bfloat16().contiguous(); K_tk = K.transpose(1,2).bfloat16().contiguous()
V_tk = V.transpose(1,2).bfloat16().contiguous(); dO_tk = dO.transpose(1,2).bfloat16().contiguous()
O_tk = out.detach().bfloat16().contiguous()
L_tk = lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()

def hk_bwd():
    dQ_in = torch.zeros((B,N,H,D),device=dev,dtype=dt).contiguous()
    dQ = torch.zeros((B,N,H,D),device=dev,dtype=dt).contiguous()
    dK = torch.zeros((B,N,H_KV,D),device=dev,dtype=dt).contiguous()
    dV = torch.zeros((B,N,H_KV,D),device=dev,dtype=dt).contiguous()
    delta = torch.zeros((B,H,N,1),device=dev).float().transpose(-1,-2).contiguous()
    tk_kernel_bkwd_prep.dispatch_prep(O_tk, dO_tk, delta)
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQ_in,dK,dV,L_tk,delta)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ)

for _ in range(WARM): hk_bwd()
t_hk=[]
for _ in range(ITERS):
    torch.cuda.synchronize(); s.record()
    hk_bwd()
    e.record(); torch.cuda.synchronize(); t_hk.append(s.elapsed_time(e))
ms_hk=sum(t_hk)/len(t_hk)

fl=flops()
print(f"config: B={B} H={H} H_KV={H_KV} N={N} D={D} causal={causal}  (Meta GQA)")
print(f"aiter bwd:  {ms_aiter:8.4f} ms   {eff(fl,ms_aiter):8.2f} TF/s")
print(f"HK    bwd:  {ms_hk:8.4f} ms   {eff(fl,ms_hk):8.2f} TF/s")
print(f"speedup HK/aiter: {ms_aiter/ms_hk:.3f}x")
