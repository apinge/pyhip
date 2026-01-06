import pyhip
import math
import torch
import torch.nn.functional as F


@pyhip.module("flash.cpp")
def forward_kernel(Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O): ...


def check_all_close(out, out_ref, rtol=0.01, atol=0.01, verbose=False):
    if not torch.allclose(out, out_ref, rtol=0.01, atol=0.01):
        if verbose:
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=0.01, atol=0.01)
        # torch.testing.assert_close(out, out_ref)
    else:
        print("PASS")


def reference_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    N: int,
    d: int,
    alpha: float,
):
    print(Q.shape)
    assert Q.shape == (M, d)
    assert K.shape == (N, d)
    assert V.shape == (N, d)
    assert output.shape == (M, d)

    scale = d**0.5
    attn = torch.matmul(Q, K.t()) / scale

    # pos_bias = alpha * (
    #     torch.arange(M, device=Q.device).unsqueeze(1)
    #     - torch.arange(N, device=K.device).unsqueeze(0)
    # )
    # attn = attn + pos_bias

    attn = torch.softmax(attn, dim=1)  # M , N
    torch.matmul(attn, V, out=output)


# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def test_flash_attn_minimal():

    kernel = forward_kernel

    dtype = torch.float32

    # Use small model params, otherwise slower than manual attention. See caveats in README.
    batch_size = 16
    n_head = 12
    seq_len = 64
    head_embd = 64

    Q = torch.randn(
        (batch_size, n_head, seq_len, head_embd), device="cuda", dtype=dtype
    )
    K = torch.randn(
        (batch_size, n_head, seq_len, head_embd), device="cuda", dtype=dtype
    )
    V = torch.randn(
        (batch_size, n_head, seq_len, head_embd), device="cuda", dtype=dtype
    )

    Bc = 32
    Br = 32
    B, nh, N, d = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
    print(f"Batch Size: {B}, Num Heads {nh}, Sequence Length{N}, Head Dim {d}")

    # Calculate tile number
    Tc = math.ceil(N / Bc)
    Tr = math.ceil(N / Br)

    softmax_scale = 1.0 / math.sqrt(d)

    # init output
    O = torch.zeros_like(Q)
    ref_output = manual_attn(Q, K, V)

    # Cumulative sum of exponents
    l = torch.zeros((B, nh, N), device=Q.device, dtype=torch.float32)

    # Init maximum of each row m: (Running Max) to negative infinity
    m = torch.full((B, nh, N), float("-inf"), device=Q.device, dtype=torch.float32)

    float_size = 4  # sizeof(float)
    sram_size = (3 * Bc * d * float_size) + (Bc * Br * float_size)

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    max_sram_size = 64 * 1024  # Current device's shared memory limit (MI308)

    print(f"Max shared memory: {max_sram_size}, requested shared memory: {sram_size}")

    if sram_size > max_sram_size:
        raise MemoryError(
            f"The requested shared memory ({sram_size} bytes) exceeds the hardware limit ({max_sram_size} bytes)! Please try reducing the size of Bc or Br."
        )
    # dim3 grid_dim(B, nh);  // batch_size x num_heads
    # dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel(
        [B, nh],
        [Bc],
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        N,
        d,
        Tc,
        Tr,
        Bc,
        Br,
        softmax_scale,
        l.data_ptr(),
        m.data_ptr(),
        O.data_ptr(),
        sharedMemBytes=sram_size,
    )
    print(O.shape)
    check_all_close(O, ref_output)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(3):
        torch.cuda._sleep(1_000_000_000)
        Q.zero_()
        ev_start.record()
        forward_kernel(
            [B, nh],
            [Bc],
            Q.data_ptr(),
            K.data_ptr(),
            V.data_ptr(),
            N,
            d,
            Tc,
            Tr,
            Bc,
            Br,
            softmax_scale,
            l.data_ptr(),
            m.data_ptr(),
            O.data_ptr(),
            sharedMemBytes=sram_size,
        )
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end) / 1
        # flops = width*height
        # bytes_per_elem = 1 # char
        # rd_bytes = (width*height* 2) * bytes_per_elem
        # print(f"dt = {dt_ms*1e3:.5f} us {flops*1e-9/dt_ms:.5f} TFLOPS  {rd_bytes*1e-6/dt_ms:.5f} GB/s per-layer  {width=} {height=} ")
        print(f"dt = {dt_ms*1e3:.5f} us ")


if __name__ == "__main__":
    test_flash_attn_minimal()
