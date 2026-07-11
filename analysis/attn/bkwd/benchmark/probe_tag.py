import torch
import tk_kernel_bkwd
import tk_kernel_bkwd_prep

# N=256 => single KV block => exactly ONE atomic contribution per output (no compounding)
B, H, H_KV, N, D = 1, 8, 1, 256, 64
dev = 'cuda'

def z(shape): return torch.zeros(shape, dtype=torch.bfloat16, device=dev)
# inputs are irrelevant (dQ_i is overwritten by the tag), just need valid buffers
Q_tk = z((B,N,H,D)); K_tk = z((B,N,H_KV,D)); V_tk = z((B,N,H_KV,D)); dO_tk = z((B,N,H,D))
L_tk = torch.zeros(B,H,1,N, dtype=torch.float32, device=dev)
delta_tk = torch.zeros(B,H,1,N, dtype=torch.float32, device=dev)
dQ_in = z((B,H,N,D))          # scratch (B,H,N,D), zero-init
dK_tk = z((B,N,H_KV,D)); dV_tk = z((B,N,H_KV,D))

tk_kernel_bkwd.dispatch_bwd_combined(Q_tk, K_tk, V_tk, dO_tk, dQ_in, dK_tk, dV_tk, L_tk, delta_tk)
torch.cuda.synchronize()
dQ_tk = z((B,N,H,D))          # natural output (B,N,H,D)
tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ_tk)
torch.cuda.synchronize()

# decode: out(Q,D) should hold tag = lane + 64*reg for the register feeding (Q,D)
# EXPECTED (per spec/my formula): lane=(D%16//4)*16 + Q, reg=D%16%4  => tag = lane + 64*reg
def expected(Q, Dc):
    dl = Dc % 16
    lane = (dl // 4) * 16 + Q
    reg = dl % 4
    return lane + 64 * reg

out = dQ_tk[0, :, 0, :].float()   # (N,D); use first 16 rows as one tile (Q=0..15)
print("Q  D | actual_tag (lane,reg) | expected_tag (lane,reg) | OK?")
bad = 0
for Q in range(16):
    for Dc in range(D):
        a = int(round(out[Q, Dc].item()))
        e = expected(Q, Dc)
        if a != e:
            bad += 1
            if Q in (0,1) and Dc < 24:
                print(f"{Q:2d} {Dc:2d} | act={a:3d} ({a%64:2d},{a//64}) | exp={e:3d} ({e%64:2d},{e//64}) | X")
print(f"total mismatches (16x{D}) = {bad}/{16*D}")
# also dump raw first row
print("Q=0 actual tags D=0..15:", [int(round(out[0,d].item())) for d in range(16)])
print("Q=0 expect tags D=0..15:", [expected(0,d) for d in range(16)])
