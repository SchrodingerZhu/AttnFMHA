# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
import math

NUM_VSM = 48

@triton.jit
def _attn_fwd_fully_static_alt(
    Q, K, V, sm_scale, Out,  #
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_vz, stride_vh, stride_vn, stride_vk,  #
    stride_oz, stride_oh, stride_om, stride_on,  #
    Z, H, N_CTX, N_CTX_K, total_tiles, num_tiles_q,  #
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr, NUM_VSM: tl.constexpr  #
):
    """
    Triton kernel for FMHA with fully static scheduling and alternating K/V direction.
    """
    pid = tl.program_id(0)
    
    # Local offsets for K/V tiles
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    local_iter = 0
    
    for global_tile_idx in range(pid, total_tiles, NUM_VSM):
        # Decompose global tile index into batch, head, and tile indices
        bh_idx = global_tile_idx // num_tiles_q
        tile_idx = global_tile_idx % num_tiles_q
        
        off_z = bh_idx // H
        off_h = bh_idx % H
        
        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh
        
        # Initialize offsets for current query tile
        offs_m = tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        
        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Load query tile
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        
        # Determine loop bounds for K/V
        if CAUSAL:
            hi = tl.minimum((tile_idx + 1) * BLOCK_M, N_CTX_K)
        else:
            hi = N_CTX_K
        
        Tc = tl.cdiv(hi, BLOCK_N)
        
        # Loop over K, V blocks with alternating direction
        for step in range(0, Tc):
            # Alternate direction based on local_iter % 2
            if local_iter % 2 < 1:
                j = step
            else:
                j = Tc - 1 - step
            
            start_n = j * BLOCK_N
            curr_n = start_n + offs_n
            
            # Load K block (transposed)
            k_ptrs = K + k_offset + offs_d[:, None] * stride_kk + curr_n[None, :] * stride_kn
            k = tl.load(k_ptrs, mask=curr_n[None, :] < N_CTX_K, other=0.0)
            
            # Compute QK
            qk = tl.dot(q, k)
            
            # Apply causal masking
            if CAUSAL:
                mask = offs_m[:, None] >= curr_n[None, :]
                qk = qk + tl.where(mask, 0.0, -1.0e6)
            
            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * sm_scale)
            qk = qk * sm_scale - m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            # Load V block and accumulate
            v_ptrs = V + v_offset + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=curr_n[:, None] < N_CTX_K, other=0.0)
            acc = tl.dot(p.to(q.dtype), v, acc)
            
            m_i = m_ij
        
        # Final normalization and store
        acc = acc / l_i[:, None]
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
        
        local_iter += 1


def triton_fmha(Q, K, V, qk_scale=None, input_pos=0, tile_m=128, tile_n=64, query_group_size=1, causal=False):
    """
    Triton FMHA with fully static scheduling and alternating K/V direction.
    """
    BLOCK_M = tile_m
    BLOCK_N = tile_n
    BLOCK_DMODEL = Q.shape[-1]
    
    if qk_scale is None:
        qk_scale = 1.0 / math.sqrt(BLOCK_DMODEL)
    
    o = torch.empty_like(Q)
    BATCH, N_HEADS, N_CTX, D_HEAD = Q.shape
    N_CTX_K = K.shape[2]
    
    num_tiles_q = triton.cdiv(N_CTX, BLOCK_M)
    total_tiles = BATCH * N_HEADS * num_tiles_q
    
    grid = (NUM_VSM, 1, 1)
    
    _attn_fwd_fully_static_alt[grid](
        Q, K, V, qk_scale, o,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, N_HEADS, N_CTX, N_CTX_K, total_tiles, num_tiles_q,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=causal, NUM_VSM=NUM_VSM,
    )
    return o

# Alias for compatibility
cutile_fmha = triton_fmha
