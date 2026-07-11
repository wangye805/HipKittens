import torch
import tk_kernel_bkwd_prep

B, H, N, D = 1, 8, 512, 64
dev = 'cuda'
NT = N // 16

def run(encode):
    # dQ_in scratch (B,H,N,D); within each 16-row tile, flat = (n%16)*64 + d
    dQ_in = torch.zeros(B, H, N, D, dtype=torch.bfloat16, device=dev)
    for n in range(N):
        r = n % 16
        for d in range(D):
            flat = r * 64 + d
            dQ_in[0, :, n, d] = float(encode(flat))
    dQ_out = torch.zeros(B, N, H, D, dtype=torch.bfloat16, device=dev)
    tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in.contiguous(), dQ_out)
    torch.cuda.synchronize()
    return dQ_out

outA = run(lambda f: f // 4)   # 0..255
outB = run(lambda f: f % 4)    # 0..3
# reconstruct flat read per output (tile0), for h=0
def myflat(Q, Dc):
    w = Dc // 16; dl = Dc % 16; qh = dl // 4; v = dl % 4
    return w*256 + (qh*16+Q)*2 + 128*(v//2) + (v%2)

print("output(Q=0,D) -> scratch_flat_read  (mine)  [* mismatch]")
mismatches = 0
for Q in range(16):
    for Dc in range(D):
        fa = int(round(outA[0, Q, 0, Dc].item()))
        fb = int(round(outB[0, Q, 0, Dc].item()))
        flat = fa*4 + fb
        mf = myflat(Q, Dc)
        if flat != mf:
            mismatches += 1
            if Q == 0:
                print(f"Q={Q} D={Dc:2d}: read={flat:4d}  mine={mf:4d}  <-- MISMATCH")
print(f"total mismatches over 16x{D} = {mismatches} / {16*D}")
# print full Q=0 row map
row0 = [(Dc, int(round(outA[0,0,0,Dc].item()))*4 + int(round(outB[0,0,0,Dc].item()))) for Dc in range(D)]
print("Q=0 (D, read_flat):", row0)
