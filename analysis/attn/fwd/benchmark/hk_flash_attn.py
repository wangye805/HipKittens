"""FlashAttention-compatible wrapper around the HipKittens hd64 forward micro-kernel.

Presents the standard `flash_attn_varlen_func` signature and adapts it to the low-level
`tk_kernel.dispatch_micro` entry point (which takes pre-allocated out/lse, a 4-D [1,total,H,D]
packed layout, and an explicit num_seqs).

Limitations of the current compiled kernel (baked at compile time), enforced by asserts:
  - causal only; head_dim = 64; softmax_scale = 1/sqrt(head_dim); bf16.
  - no dropout / SWA (window_size) / softcap / alibi.
  - the .so is specialized for a fixed (ATTN_H, ATTN_H_KV): H/H_KV of the inputs must match the
    kernel it was built with (a production build would compile a set and dispatch by config).
Supported (no caller obligations): arbitrary (non-aligned) cu_seqlens, cross-attn (cu_seqlens_q !=
cu_seqlens_k), dynamic batch, no tail padding.
"""
import math
import torch
import tk_kernel

HEAD_DIM = 64  # ATTN_D the kernel is compiled with


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
):
    # --- feature gates (what the compiled kernel supports) ---
    assert causal, "HK hd64 fwd is causal-only"
    assert dropout_p == 0.0, "dropout not supported"
    assert tuple(window_size) == (-1, -1), "sliding-window attention (SWA) not supported yet"
    assert softcap == 0.0, "softcap not supported"
    assert alibi_slopes is None, "alibi not supported"
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
        "expected packed varlen layout q:[total_q,H,D], k/v:[total_k,H_kv,D]"
    total_q, H, D = q.shape
    total_k, H_kv, Dk = k.shape
    assert D == HEAD_DIM and Dk == HEAD_DIM, f"head_dim must be {HEAD_DIM}"
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, "bf16 only"
    if softmax_scale is not None:
        assert abs(softmax_scale - 1.0 / math.sqrt(HEAD_DIM)) < 1e-6, \
            "custom softmax_scale not supported (kernel bakes 1/sqrt(head_dim))"

    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
    num_seqs = cu_seqlens_q.numel() - 1

    out = torch.empty((total_q, H, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((1, H, 1, total_q), dtype=torch.float32, device=q.device)  # kernel's L_vec layout

    tk_kernel.dispatch_micro(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),   # -> [1,total,H,D] gl views (share storage)
        out.unsqueeze(0), lse,
        cu_seqlens_q, cu_seqlens_k,
        int(max_seqlen_q), int(num_seqs),
    )
    if return_lse:
        return out, lse[0, :, 0, :]   # [H, total_q], FA-style per-(head,token) LSE
    return out
