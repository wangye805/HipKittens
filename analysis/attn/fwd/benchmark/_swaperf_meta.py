import torch, sys, itertools, tk_kernel
# Timed HK SWA forward at Meta's 5.5B shape for absolute-ms comparison vs H100 FA3.
# Cross-attn (Sq may differ from Skv), bottom-right causal window. H/Hkv fixed by the build.
# argv: Sq Skv W B [H Hkv]
Sq = int(sys.argv[1]); Skv = int(sys.argv[2]); W = int(sys.argv[3]); B = int(sys.argv[4])
H  = int(sys.argv[5]) if len(sys.argv) > 5 else 128
Hk = int(sys.argv[6]) if len(sys.argv) > 6 else 16
D = 64; dt = torch.bfloat16; torch.manual_seed(0)
cuq = torch.tensor([0] + list(itertools.accumulate([Sq]*B)),  dtype=torch.int32, device='cuda')
cuk = torch.tensor([0] + list(itertools.accumulate([Skv]*B)), dtype=torch.int32, device='cuda')
totq = int(cuq[-1]); totk = int(cuk[-1])
q = torch.randn(totq, H,  D, dtype=dt, device='cuda')
k = torch.randn(totk, Hk, D, dtype=dt, device='cuda')
v = torch.randn(totk, Hk, D, dtype=dt, device='cuda')
out = torch.zeros(1, totq, H, D, dtype=dt, device='cuda')
lse = torch.zeros(1, H, 1, totq, dtype=torch.float32, device='cuda')
qp, kp, vp = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
args = (qp, kp, vp, out, lse, cuq, cuk, Sq, B, W, 0)
for _ in range(30): tk_kernel.dispatch_micro(*args)
torch.cuda.synchronize()
st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
st.record()
for _ in range(100): tk_kernel.dispatch_micro(*args)
en.record(); torch.cuda.synchronize()
ms = st.elapsed_time(en) / 100
print(f"[HK SWA meta] Sq={Sq} Skv={Skv} W={W} B={B} H={H}/{Hk}  {ms:.4f} ms")
