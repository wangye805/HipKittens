"""Time HK GQA causal backward, det vs non-det build. Square, D=128.

  python bench_hk_bwd.py B N H Hkv causal mode      # mode = det | nondet
The build must match the mode (det => built with HK_DETERMINISTIC=1).
"""
import sys, torch
import tk_kernel_bkwd, tk_kernel_bkwd_prep  # HK fwd doesn't build on ROCm7.2; use torch O/L

torch.manual_seed(0)
dev = "cuda"
B   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H   = int(sys.argv[3]) if len(sys.argv) > 3 else 64
HKV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
causal = bool(int(sys.argv[5])) if len(sys.argv) > 5 else True
mode = sys.argv[6] if len(sys.argv) > 6 else "nondet"
D = 128
BLOCK_KV = 256
nsplit = N // BLOCK_KV
warmup, iters = 50, 100

def gen(s): return torch.randn(s, dtype=torch.bfloat16, device=dev)
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = gen((B,H,N,D)), gen((B,HKV,N,D)), gen((B,HKV,N,D)), gen((B,H,N,D))
Q_tk  = Q_bhnd.transpose(1,2).contiguous()
K_tk  = K_bhnd.transpose(1,2).contiguous()
V_tk  = V_bhnd.transpose(1,2).contiguous()
dO_tk = dO_bhnd.transpose(1,2).contiguous()
# O/L from a quick torch reference (timing-representative; exact values don't affect timing)
group = H // HKV; scale = D ** -0.5
Ke = K_bhnd.float().repeat_interleave(group,1); Ve = V_bhnd.float().repeat_interleave(group,1)
S = (Q_bhnd.float() @ Ke.transpose(-1,-2)) * scale
if causal: S = S.masked_fill(torch.triu(torch.ones(N,N,device=dev,dtype=torch.bool),1), float("-inf"))
lse = torch.logsumexp(S, -1)
O_tk = (torch.softmax(S,-1) @ Ve).transpose(1,2).bfloat16().contiguous()  # [B,N,H,D]
L_tk = lse.reshape(B,H,1,N).float().contiguous()                          # [B,H,1,N]
del S, Ke, Ve

def alloc():
    return (torch.zeros((B,H,N,D), dtype=torch.bfloat16, device=dev).contiguous(),
            torch.zeros((B,N,H,D), dtype=torch.bfloat16, device=dev).contiguous(),
            torch.zeros((B,N,HKV,D), dtype=torch.bfloat16, device=dev).contiguous(),
            torch.zeros((B,N,HKV,D), dtype=torch.bfloat16, device=dev).contiguous(),
            torch.zeros((B,H,N,1), device=dev).float().transpose(-1,-2).contiguous())

dQ_in, dQ, dK, dV, delta = alloc()
dQ_acc = torch.zeros((nsplit*B, H, N, D), dtype=torch.float32, device=dev).contiguous() if mode=="det" else None

def time_op(fn, label):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sum(ts)/len(ts)

def bwd():
    if mode=="det":
        tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQ_in,dK,dV,L_tk,delta,dQ_acc)
    else:
        tk_kernel_bkwd.dispatch_bwd_combined(Q_tk,K_tk,V_tk,dO_tk,dQ_in,dK,dV,L_tk,delta)

t_bwd = time_op(bwd, "bwd")
f = 4 * B * N*N * H * D // (2 if causal else 1)
tf_bwd = (2.5*f/1e12) / (t_bwd/1e3)
line = f"[{mode}] B={B} N={N} H={H} Hkv={HKV} D={D} nsplit={nsplit} | bwd-only {t_bwd:.3f} ms ({tf_bwd:.0f} TF/s)"
if mode=="det":
    t_red = time_op(lambda: tk_kernel_bkwd_prep.dispatch_dq_reduce(dQ_acc, dQ_in), "reduce")
    line += f" | reduce {t_red:.3f} ms | bwd+reduce {t_bwd+t_red:.3f} ms"
print(line)
