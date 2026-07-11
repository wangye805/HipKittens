import torch
import aiter
import tk_kernel_bkwd

B, H, H_KV, N, D = 1, 8, 1, 512, 64
dev, dt = 'cuda', torch.bfloat16
NTILE = N // 16

def gen(shape):
    t = torch.randn(shape, dtype=dt, device=dev)
    mag = torch.norm(t, dim=-1, keepdim=True)
    return (t * (torch.randn(mag.shape, dtype=dt, device=dev)*0.1 + 10)/mag).contiguous()

scr_samples = []   # each (nsamp, 1024) scratch block flattened
nat_samples = []
NSEED = 20
for seed in range(NSEED):
    torch.manual_seed(seed+1)
    Q = gen((B,H,N,D)); K = gen((B,H_KV,N,D)); V = gen((B,H_KV,N,D)); dO = gen((B,H,N,D))
    Qa = Q.transpose(1,2).contiguous().detach().requires_grad_(True)
    Ka = K.transpose(1,2).contiguous().detach().requires_grad_(True)
    Va = V.transpose(1,2).contiguous().detach().requires_grad_(True)
    dOa = dO.transpose(1,2).contiguous()
    out, lse = aiter.flash_attn_func(Qa, Ka, Va, causal=True, return_lse=True, deterministic=False)
    out.backward(dOa)
    dQ_ref = Qa.grad                                    # (B,N,H,D)
    Q_tk = Q.transpose(1,2).bfloat16().contiguous(); K_tk = K.transpose(1,2).bfloat16().contiguous()
    V_tk = V.transpose(1,2).bfloat16().contiguous(); dO_tk = dO.transpose(1,2).bfloat16().contiguous()
    O_tk = out.detach().bfloat16().contiguous()
    L_tk = lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()
    delta = (dO_tk.float()*O_tk.float()).sum(-1)
    delta_tk = delta.permute(0,2,1).unsqueeze(2).contiguous()
    dQ_in = torch.zeros_like(Qa.grad).bfloat16().transpose(1,2).contiguous()
    dK_tk = torch.zeros_like(Ka.grad).bfloat16().contiguous(); dV_tk = torch.zeros_like(Va.grad).bfloat16().contiguous()
    tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)
    torch.cuda.synchronize()
    for b in range(B):
        for qh in range(H):
            for T in range(NTILE):
                scr_samples.append(dQ_in[b, qh, T*16:T*16+16, :].float().reshape(-1))
                nat_samples.append(dQ_ref[b, T*16:T*16+16, qh, :].float().reshape(-1))

scr = torch.stack(scr_samples)   # (S, 1024)
nat = torch.stack(nat_samples)
print(f"samples={scr.shape[0]}")
sn = scr - scr.mean(0,keepdim=True); sn = sn/(sn.norm(dim=0,keepdim=True)+1e-9)
nn = nat - nat.mean(0,keepdim=True); nn = nn/(nn.norm(dim=0,keepdim=True)+1e-9)
C = nn.t() @ sn                  # (1024 nat, 1024 scratch)
best = C.max(dim=1)
mp = best.indices
print(f"mean best corr: {best.values.mean().item():.4f}  frac>0.999: {(best.values>0.999).float().mean().item():.3f}")
# print map for nat row0 (Q=0), all D, as scratch flat, plus my-formula prediction
def myflat(Q,Dc):
    w=Dc//16; dl=Dc%16; qh=dl//4; v=dl%4
    return w*256 + (qh*16+Q)*2 + 128*(v//2) + (v%2)
print("D : true_flat (my_flat) corr  [* = mismatch]")
for Dc in range(D):
    tf = mp[0*D+Dc].item(); mf = myflat(0,Dc); cc = best.values[0*D+Dc].item()
    mark = "" if tf==mf else "  <-- MISMATCH"
    print(f"D={Dc:2d}: true={tf:4d} mine={mf:4d} corr={cc:.3f}{mark}")
