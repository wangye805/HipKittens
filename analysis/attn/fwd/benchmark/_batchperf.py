import torch, sys, tk_kernel
# Batch (non-varlen) HK forward timing, HK-only (no aiter ref). Square causal.
# Kernel is compiled for fixed ATTN_N/B/H/H_KV; pass matching B N H Hk here.
B = int(sys.argv[1]); N = int(sys.argv[2])
H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
Hk = int(sys.argv[4]) if len(sys.argv) > 4 else 16
D = 64; dt = torch.bfloat16; torch.manual_seed(0)
q = torch.randn(B, N, H,  D, dtype=dt, device='cuda')
k = torch.randn(B, N, Hk, D, dtype=dt, device='cuda')
v = torch.randn(B, N, Hk, D, dtype=dt, device='cuda')
out = torch.zeros(B, N, H, D, dtype=dt, device='cuda')
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
for _ in range(30): tk_kernel.dispatch_micro(q, k, v, out, lse)
torch.cuda.synchronize()
st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
st.record()
for _ in range(100): tk_kernel.dispatch_micro(q, k, v, out, lse)
en.record(); torch.cuda.synchronize()
ms = st.elapsed_time(en) / 100
flop = 2 * B * N * N * H * D          # causal square = 4*B*N^2*H*D/2
tf = flop / 1e12 / (ms / 1e3)
print(f"[HK batch] B={B} N={N} H={H}/{Hk}  {ms:.4f} ms  {tf:.1f} TF/s")
