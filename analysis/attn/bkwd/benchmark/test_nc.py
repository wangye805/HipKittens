import torch, os
import aiter
import tk_kernel_bkwd
import tk_kernel_bkwd_prep

torch.manual_seed(0)
N = int(os.environ.get("NN", "512"))
B, H, H_KV, D = 1, 8, 1, int(os.environ.get("DD","64"))
dev, dt = 'cuda', torch.bfloat16

def robust(ref, pred):
    ref, pred = ref.float(), pred.float()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    l2 = ((ref-pred).pow(2).sum().sqrt() / ref.pow(2).sum().sqrt().clamp_min(1e-9)).item()
    return cos, l2

def gen(shape):
    t = torch.randn(shape, dtype=dt, device=dev)
    mag = torch.norm(t, dim=-1, keepdim=True)
    return (t * (torch.randn(mag.shape, dtype=dt, device=dev)*0.1 + 10)/mag).contiguous()

Q = gen((B,H,N,D)); K = gen((B,H_KV,N,D)); V = gen((B,H_KV,N,D)); dO = gen((B,H,N,D))
Qa = Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka = K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va = V.transpose(1,2).contiguous().detach().requires_grad_(True)
dOa = dO.transpose(1,2).contiguous()
out, lse = aiter.flash_attn_func(Qa, Ka, Va, causal=False, return_lse=True, deterministic=False)
out.backward(dOa)
dQ_ref, dK_ref, dV_ref = Qa.grad, Ka.grad, Va.grad

Q_tk = Q.transpose(1,2).bfloat16().contiguous(); K_tk = K.transpose(1,2).bfloat16().contiguous()
V_tk = V.transpose(1,2).bfloat16().contiguous(); dO_tk = dO.transpose(1,2).bfloat16().contiguous()
O_tk = out.detach().bfloat16().contiguous()
L_tk = lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()
delta = (dO_tk.float()*O_tk.float()).sum(-1)
delta_tk = delta.permute(0,2,1).unsqueeze(2).contiguous()
dQ_in = torch.zeros_like(Qa.grad).bfloat16().transpose(1,2).contiguous()
dK_tk = torch.zeros_like(dK_ref).bfloat16().contiguous(); dV_tk = torch.zeros_like(dV_ref).bfloat16().contiguous()
tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)
torch.cuda.synchronize()
dQ_tk = torch.zeros_like(dQ_ref).bfloat16().contiguous()
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ_tk)
torch.cuda.synchronize()

cv,l2v = robust(dV_ref,dV_tk); ck,l2k = robust(dK_ref,dK_tk); cq,l2q = robust(dQ_ref,dQ_tk)
print(f"N={N}  dV: cos={cv:.6f}  dK: cos={ck:.6f}  dQ: cos={cq:.6f} rel_l2={l2q:.4f}")

# atomic-vs-shuffle: does scratch dQ_in hold correct values (layout-independent)?
si=dQ_in.flatten().float().sort().values; sr=dQ_ref.flatten().float().sort().values
cs=torch.nn.functional.cosine_similarity(si,sr,dim=0).item()
print(f"[scratch] dQ_in vs dQ_ref sorted(multiset): cos={cs:.6f}  (~1 => atomic OK, shuffle transposes wrong)")

# per-D-col cos (is it the cols=0,1 mod4 transpose_2d<1,1> pattern?)
a=dQ_tk[0].float().reshape(-1,D); b=dQ_ref[0].float().reshape(-1,D)
an=a/(a.norm(dim=0,keepdim=True)+1e-9); bn=b/(b.norm(dim=0,keepdim=True)+1e-9)
d=(an*bn).sum(0)
print("per-col cos:", " ".join(f"{d[i]:.2f}" for i in range(32)))
