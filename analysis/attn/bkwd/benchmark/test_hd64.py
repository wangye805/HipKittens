import torch, math
import aiter
import tk_kernel_bkwd        # our hd64 bwd (built small: B=1 H=8 H_KV=1 N=512 D=64, causal)
import tk_kernel_bkwd_prep   # prep: dispatch_dq_shuffle (for dQ)

torch.manual_seed(0)
B, H, H_KV, N, D = 1, 8, 1, 512, 64
causal = True
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

# BHND gen
Q = gen((B,H,N,D)); K = gen((B,H_KV,N,D)); V = gen((B,H_KV,N,D)); dO = gen((B,H,N,D))

# ---- aiter: reference grads + O + LSE (BNHD) ----
Qa = Q.transpose(1,2).contiguous().detach().requires_grad_(True)
Ka = K.transpose(1,2).contiguous().detach().requires_grad_(True)
Va = V.transpose(1,2).contiguous().detach().requires_grad_(True)
dOa = dO.transpose(1,2).contiguous()
out, lse = aiter.flash_attn_func(Qa, Ka, Va, causal=causal, return_lse=True, deterministic=False)
out.backward(dOa)
dQ_ref, dK_ref, dV_ref = Qa.grad, Ka.grad, Va.grad      # BNHD
print("shapes: out", tuple(out.shape), "lse", tuple(lse.shape))

# ---- HK bwd inputs (BNHD), mirroring test_python.py conventions ----
Q_tk = Q.transpose(1,2).bfloat16().contiguous()
K_tk = K.transpose(1,2).bfloat16().contiguous()
V_tk = V.transpose(1,2).bfloat16().contiguous()
dO_tk = dO.transpose(1,2).bfloat16().contiguous()
O_tk = out.detach().bfloat16().contiguous()
# L_tk (B,H,1,N)  [assume aiter lse is (B,H,N); adjust if the printed shape differs]
L_tk = lse.detach().reshape(B,H,N).unsqueeze(-1).transpose(-1,-2).float().contiguous()
# delta (B,H,1,N) = rowsum_D(dO * O)
delta = (dO_tk.float()*O_tk.float()).sum(-1)            # (B,N,H)
delta_tk = delta.permute(0,2,1).unsqueeze(2).contiguous()   # (B,H,1,N)

dQ_in = torch.zeros_like(Qa.grad).bfloat16().transpose(1,2).contiguous()
dK_tk = torch.zeros_like(dK_ref).bfloat16().contiguous()
dV_tk = torch.zeros_like(dV_ref).bfloat16().contiguous()

tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)
torch.cuda.synchronize()

# ---- dQ: shuffle dQ_in -> dQ, compare ----
dQ_tk = torch.zeros_like(dQ_ref).bfloat16().contiguous()   # BNHD (B,N,H,D)
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ_tk)
torch.cuda.synchronize()

# --- diagnostic: find the exact permutation dQ_tk vs dQ_ref on D and N axes ---
def corr_perm(a, b, axis_len, name):
    # a,b: (..., axis_len). flatten all but last dim; find best-match column mapping
    A = a.reshape(-1, axis_len).float()
    B = b.reshape(-1, axis_len).float()
    An = A / (A.norm(dim=0, keepdim=True) + 1e-9)
    Bn = B / (B.norm(dim=0, keepdim=True) + 1e-9)
    C = An.t() @ Bn                       # C[i,j]=cos(tk col i, ref col j)
    perm = C.argmax(dim=1)
    diagfrac = (perm == torch.arange(axis_len, device=perm.device)).float().mean().item()
    maxcorr = C.max(dim=1).values.mean().item()
    print(f"[perm] {name}: diag_frac={diagfrac:.3f} maxcorr={maxcorr:.4f}")
    if name == "D-axis":
        d = C.diagonal()
        print("[perm] D per-col self-cos: " +
              " ".join(f"{d[i]:.2f}" for i in range(axis_len)))
corr_perm(dQ_tk, dQ_ref, D, "D-axis")                     # (B,N,H,D) last axis = D
corr_perm(dQ_tk.transpose(1,3), dQ_ref.transpose(1,3), dQ_ref.shape[1], "N-axis")

# --- element-level scramble within one 16xD dot-slice block (b=0,h=0,rows0:16) ---
tk = dQ_tk[0, 0:16, 0, :].float().reshape(-1)   # 16*D
rf = dQ_ref[0, 0:16, 0, :].float().reshape(-1)
diff = (tk[:, None] - rf[None, :]).abs()
match = diff.argmin(dim=1)
ident = (match == torch.arange(tk.numel(), device=tk.device)).float().mean().item()
print(f"[blk] within-block identity frac={ident:.3f}  (elem i at r=i//{D},c=i%{D})")
mism = [(i//D, i%D, match[i].item()//D, match[i].item()%D)
        for i in range(tk.numel()) if match[i].item()!=i][:24]
print(f"[blk] mismatches (tr,tc)->(rr,rc): {mism}")

# ============================================================================
# (ii) Nail the scratch->natural map in Python via correlation over blocks.
#   scratch dQ_in: (B,H,N,D). Main kernel writes a dot-slice's 16x64 dQ into a
#   1024-flat region at coord {b,qh,r0,0}, r0 = q_seq*4+ds, flat = warpid*256 + ...
#   natural dQ_ref: (B,N,H,D).  A dot-slice = 16 consecutive Q rows.
#   We treat each (b, qh, dot-slice) as ONE sample of a 1024-vector on each side
#   and find, per natural position, the best-correlated scratch flat position.
# ============================================================================
Bd,Hd,Nd,Dd = dQ_in.shape                       # (1,8,512,64)
nblk = Nd // 16
# scratch samples: each 16-row region flattened to 1024 (row-major: row*64+d)
scr = dQ_in.reshape(Bd, Hd, nblk, 16*Dd).float()          # (B,H,nblk,1024)
scr = scr.permute(0,1,2,3).reshape(-1, 16*Dd)             # (samples,1024)
# natural samples: dQ_ref (B,N,H,D) -> (B,H,nblk,16,D) -> (samples,1024)
nat = dQ_ref.permute(0,2,1,3).reshape(Bd,Hd,nblk,16,Dd).reshape(-1,16*Dd).float()  # (samples,1024)
# correlate columns
sn = scr - scr.mean(0, keepdim=True); sn = sn/(sn.norm(dim=0,keepdim=True)+1e-9)
nn = nat - nat.mean(0, keepdim=True); nn = nn/(nn.norm(dim=0,keepdim=True)+1e-9)
Cmap = nn.t() @ sn                              # (1024 nat, 1024 scratch)
best = Cmap.max(dim=1)
mp = best.indices                               # nat pos -> scratch flat
print(f"[map] mean best-corr over 1024 nat positions: {best.values.mean().item():.4f}")
print(f"[map] positions with corr>0.99: {(best.values>0.99).float().mean().item():.3f}")
# show the map for natural row 0, all 64 D (nat pos = 0*64+d = d), as (scr_row,scr_col)
row0 = [(mp[d].item()//Dd, mp[d].item()%Dd) for d in range(Dd)]
print(f"[map] nat (row0,d)->scratch(row,col) for d=0..15: {row0[:16]}")
print(f"[map] nat (row0,d)->scratch(row,col) for d=16..31: {row0[16:32]}")

# --- diagnostic: is the PRE-SHUFFLE scratch (dQ_in) value-correct (just permuted)? ---
sin = dQ_in.flatten().float().sort().values
sref = dQ_ref.flatten().float().sort().values
csort = torch.nn.functional.cosine_similarity(sin, sref, dim=0).item()
print(f"[diag] dQ_in vs dQ_ref sorted (perm-invariant): cos={csort:.6f}  "
      f"(~1 => atomic/compute OK, shuffle wrong; <1 => atomic/compute wrong)")
_nzi = (dQ_in.float().abs()>1e-9).float().mean().item()
_nzr = (dQ_ref.float().abs()>1e-9).float().mean().item()
print(f"[diag] dQ_in nonzero frac={_nzi:.3f}  dQ_ref nonzero frac={_nzr:.3f}")

# --- DECISIVE: reconstruct dQ from scratch via my derived flat(Q,D) formula ---
#   flat(Q,D) = w*256 + (qh*16+Q)*2 + 128*(v//2) + (v%2),  w=D//16, dl=D%16, qh=dl//4, v=dl%4
blk = dQ_in[0, 0, 0:16, :].float().reshape(-1)     # first output tile scratch (1024)
rec = torch.zeros(16, D)
for Q in range(16):
    for Dc in range(D):
        w = Dc // 16; dl = Dc % 16; qh = dl // 4; v = dl % 4
        flat = w*256 + (qh*16 + Q)*2 + 128*(v//2) + (v%2)
        rec[Q, Dc] = blk[flat]
refblk = dQ_ref[0, 0:16, 0, :].float()             # natural first tile (16,D)
rc = torch.nn.functional.cosine_similarity(rec.flatten(), refblk.flatten(), dim=0).item()
print(f"[recon] my-formula reconstruct vs ref (tile0): cos={rc:.6f}")
# per-D-col of the reconstruction
dcol = [torch.nn.functional.cosine_similarity(rec[:,d], refblk[:,d], dim=0).item() for d in range(D)]
print("[recon] per-col cos: " + " ".join(f"{x:.2f}" for x in dcol[:32]))

cv, l2v = robust(dV_ref, dV_tk)
ck, l2k = robust(dK_ref, dK_tk)
cq, l2q = robust(dQ_ref, dQ_tk)
print(f"dV: cos={cv:.6f}  rel_l2={l2v:.4f}")
print(f"dK: cos={ck:.6f}  rel_l2={l2k:.4f}")
print(f"dQ: cos={cq:.6f}  rel_l2={l2q:.4f}")
print("dQ HK :", dQ_tk.flatten()[:6].float().tolist())
print("dQ ref:", dQ_ref.flatten()[:6].float().tolist())
print("VERDICT:",
      "ALL PASS" if (cv>0.99 and ck>0.99 and cq>0.99)
      else ("dV/dK PASS, dQ FAIL (K_j_col swizzle?)" if (cv>0.99 and ck>0.99) else "FRONT-END/dV/dK FAIL"))
