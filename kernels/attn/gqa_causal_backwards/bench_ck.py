"""CK/aiter flash-attn bwd timing on the SAME shape as the HK bench, same GPU.
  python bench_ck.py 4 4096 64 8 1
Times .backward() (full bwd incl. delta/reduce), det=False and det=True.
"""
import sys, torch
from flash_attn import flash_attn_func
torch.manual_seed(0)
dev = "cuda"
B   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H   = int(sys.argv[3]) if len(sys.argv) > 3 else 64
HKV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
causal = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
D = 128
warmup, iters = 50, 100
def gen(s, rg): return torch.randn(s, dtype=torch.bfloat16, device=dev, requires_grad=rg)
Q = gen((B,N,H,D), True); K = gen((B,N,HKV,D), True); V = gen((B,N,HKV,D), True)
dO = torch.randn((B,N,H,D), dtype=torch.bfloat16, device=dev)
f = 4 * B * N*N * H * D // (2 if causal else 1)

def call(det):
    o = flash_attn_func(Q, K, V, causal=causal, deterministic=det)
    return o[0] if isinstance(o, tuple) else o

def bench(det):
    try:
        for _ in range(warmup):
            Q.grad = K.grad = V.grad = None
            call(det).backward(dO)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            Q.grad = K.grad = V.grad = None
            o = call(det)
            torch.cuda.synchronize(); s.record()
            o.backward(dO)
            e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        ms = sum(ts)/len(ts)
        print(f"[CK/flash_attn det={det}] B={B} N={N} H={H} Hkv={HKV} D={D} | bwd {ms:.3f} ms | {2.5*f/1e12/(ms/1e3):.0f} TF/s")
    except Exception as ex:
        print(f"[CK/flash_attn det={det}] FAILED: {ex}")

bench(False)
bench(True)
