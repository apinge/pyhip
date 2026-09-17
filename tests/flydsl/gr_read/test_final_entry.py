"""Final T1..32 pipeline checks using plain assert torch.allclose.

Direct: python3 test_final_entry.py --rows 1 17 --graph
Debug:  python3 test_final_entry.py --rows 1 --debug
All:    python3 test_final_entry.py --graph --check-contract

T1..16 selects SmallBatchGRRead; T17..32 selects LargeDownGRRead.
Both write four FP32 linear partials before the up kernel applies SiLU.
"""

import argparse

import torch
import torch.nn.functional as F

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .large_down import LargeDownGRRead
    from .prefetch_up import MAX_ROWS
    from .small_batch import SmallBatchGRRead
else:
    from combined_host import CombinedPaddedGRRead
    from large_down import LargeDownGRRead
    from prefetch_up import MAX_ROWS
    from small_batch import SmallBatchGRRead

C, H, R = 4, 2560, 320
K = C * H
OUTPUT_TOLERANCE = dict(rtol=1e-2, atol=5e-3)
DOWN_TOLERANCE = dict(rtol=2e-5, atol=2e-5)


def make_case(rows, seed=131):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16, generator=generator)
    w_down = torch.randn(R, K, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    w_up = torch.randn(K, R, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.02
    return x, w_down, w_up


def reference_fp64(x, w_down, w_up):
    """Use original weights and stream-major X, independent of kernel packing."""
    x64 = x.double()
    activation = F.silu((x64 @ w_down.double().T) / C)
    gates = torch.sigmoid(activation @ w_up.double().T).reshape(x.shape[0], C, H)
    return (gates * x64.reshape(x.shape[0], C, H)).mean(dim=1)


def prepare_reader(rows, w_down, w_up):
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"final pipeline requires T=1..{MAX_ROWS}")
    # Only use the control constructor to prepare the unchanged packed weights.
    prepared = CombinedPaddedGRRead(rows, w_down, w_up)
    reader = SmallBatchGRRead(prepared) if rows <= 16 else LargeDownGRRead(prepared)
    assert reader.w_down.data_ptr() == prepared.w_down.data_ptr()
    assert reader.w_up.data_ptr() == prepared.w_up.data_ptr()
    assert reader.partial.data_ptr() != prepared.partial.data_ptr()
    assert reader.output.data_ptr() != prepared.output.data_ptr()
    return reader


@torch.inference_mode()
def check_case(rows, debug=False):
    x, w_down, w_up = make_case(rows)
    reader = prepare_reader(rows, w_down, w_up)
    down_config = reader.config.down if rows <= 16 else reader.config
    up_config = reader.up_config
    padded_rows = 16 if rows <= 16 else 32
    assert down_config.block_m == padded_rows
    assert down_config.block_k == 128
    assert down_config.waves == 4
    assert down_config.global_split == 4
    assert down_config.prefetch and down_config.interleave
    assert down_config.prefetch_unroll == (2 if rows <= 25 else 1)
    assert up_config.down_mode == "partial"
    assert up_config.split_k == 4
    assert up_config.block_m == 16
    assert up_config.up_n == (64 if 7 <= rows <= 15 else 128)
    assert up_config.waves == 4
    assert up_config.hidden_pad == 4
    assert up_config.prefetch_low is False
    assert up_config.compensate_hidden is True
    assert up_config.preshuffle is True
    assert reader.partial.shape == (4 * padded_rows * R,)
    assert reader.partial.dtype == torch.float32
    assert reader.output.shape == (rows, H)
    assert reader.output.dtype == torch.bfloat16
    packed_down = reader.w_down.clone()
    packed_up = reader.w_up.clone()

    if debug:
        breakpoint()

    # Clone: all calls on this reader reuse its output buffer.
    combined = reader(x).clone()
    torch.cuda.synchronize(x.device)
    expected = reference_fp64(x, w_down, w_up)
    assert torch.allclose(combined.double(), expected, **OUTPUT_TOLERANCE), f"T={rows}: combined vs FP64"

    reader.partial.fill_(float("nan"))
    reader.run_down(x)
    torch.cuda.synchronize(x.device)
    down_fp64 = x.double() @ w_down.double().T
    partials = reader.partial.reshape(4, padded_rows, R)
    down_actual = partials.sum(dim=0)[:rows]
    assert torch.isfinite(partials).all(), f"T={rows}: down must overwrite poisoned P"
    assert torch.allclose(down_actual.double(), down_fp64, **DOWN_TOLERANCE), f"T={rows}: linear down vs FP64"
    assert torch.all(partials[:, rows:, :] == 0), f"T={rows}: down padding must be zero"

    reader.run_up(x)
    torch.cuda.synchronize(x.device)
    assert torch.allclose(reader.output.double(), expected, **OUTPUT_TOLERANCE), f"T={rows}: staged output vs FP64"
    assert torch.allclose(reader.output, combined, **OUTPUT_TOLERANCE), f"T={rows}: staged vs combined output"
    assert torch.equal(reader.w_down, packed_down), f"T={rows}: W_down must not change"
    assert torch.equal(reader.w_up, packed_up), f"T={rows}: W_up must not change"
    return {
        "rows": rows,
        "entry": type(reader).__name__,
        "down": "down_wave_splitk_pipeline",
        "prefetch_unroll": down_config.prefetch_unroll,
        "up": "up_gate",
        "up_n": up_config.up_n,
        "partial_shape": tuple(partials.shape),
        "partial_elements": reader.partial.numel(),
        "down_max_abs": (down_actual.double() - down_fp64).abs().max().item(),
        "output_max_abs": (combined.double() - expected).abs().max().item(),
    }


@torch.inference_mode()
def check_graph_case(rows):
    x, w_down, w_up = make_case(rows, seed=137)
    reader = prepare_reader(rows, w_down, w_up)
    for _ in range(3):
        reader(x)
    torch.cuda.synchronize(x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = reader(x)
    for replay in range(3):
        x.mul_(0.9).add_(0.03125)
        reader.partial.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(x.device)
        expected = reference_fp64(x, w_down, w_up)
        assert torch.allclose(actual.double(), expected, **OUTPUT_TOLERANCE), f"T={rows}: graph replay {replay + 1}"
        assert torch.isfinite(reader.partial).all(), f"T={rows}: replay must overwrite P"
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(actual) == 0, f"T={rows}: zero input must produce zero output"
    assert torch.count_nonzero(reader.partial) == 0, f"T={rows}: zero input must produce zero partials"


@torch.inference_mode()
def check_input_contract():
    _, w_down, w_up = make_case(1)
    for invalid_rows in (-1, 0, MAX_ROWS + 1):
        try:
            prepare_reader(invalid_rows, w_down, w_up)
        except ValueError as error:
            assert f"1..{MAX_ROWS}" in str(error), str(error)
        else:
            assert False, f"T={invalid_rows}: expected ValueError"

    for rows in (1, 17):
        x, w_down, w_up = make_case(rows)
        reader = prepare_reader(rows, w_down, w_up)
        for invalid_x, expected_message in (
            (x.float(), "prepared"),
            (x.repeat(2, 1), "prepared"),
            (x.cpu(), "prepared"),
            (torch.empty(rows, K * 2, dtype=x.dtype, device=x.device)[:, ::2], "contiguous"),
        ):
            try:
                reader(invalid_x)
            except ValueError as error:
                assert expected_message in str(error), str(error)
            else:
                assert False, f"shape={invalid_x.shape}, dtype={invalid_x.dtype}: expected ValueError ({expected_message})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(1, MAX_ROWS + 1)))
    parser.add_argument("--graph", action="store_true", help="also verify changed-input graph replay")
    parser.add_argument("--check-contract", action="store_true", help="also check rejection of empty/invalid inputs")
    parser.add_argument(
        "--debug", action="store_true", help="break after reader construction, before the real input call"
    )
    args = parser.parse_args()
    if any(not 1 <= rows <= MAX_ROWS for rows in args.rows):
        parser.error(f"--rows must be in 1..{MAX_ROWS}; use --check-contract to check empty and invalid inputs")
    for rows in args.rows:
        print(check_case(rows, debug=args.debug), flush=True)
        if args.graph:
            check_graph_case(rows)
            print(f"T={rows}: changed-input graph replay passed", flush=True)
    if args.check_contract:
        check_input_contract()
        print("Final-entry empty/invalid input rejection checks passed", flush=True)
