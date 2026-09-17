"""Plain Python correctness/debug checks using assert torch.allclose.

Direct: python3 test_final_entry.py --rows 1 17 --graph
Debug:  python3 test_final_entry.py --rows 1 --debug
All:    python3 test_final_entry.py --graph --check-contract
"""

import argparse

import torch
import torch.nn.functional as F

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .prefetch_up import MAX_ROWS
else:
    from combined_host import CombinedPaddedGRRead
    from prefetch_up import MAX_ROWS

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


def check_case(rows, debug=False):
    x, w_down, w_up = make_case(rows)
    reader = CombinedPaddedGRRead(rows, w_down, w_up)
    expected_mode = "partial" if rows <= 16 else "wave_splitk"
    assert reader.config.down_mode == expected_mode
    assert reader.config.hidden_pad == 4
    assert reader.config.prefetch_low is False
    assert reader.config.compensate_hidden is True
    assert reader.config.preshuffle is True

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
    if expected_mode == "partial":
        partials = reader.partial.reshape(reader.config.split_k, -1, R)
        down_actual = partials.sum(dim=0)[:rows]
        down_expected = down_fp64
        padding = partials[:, rows:, :]
        down_name = "down_partial"
    else:
        activation = reader.partial.reshape(-1, R)
        down_actual = activation[:rows]
        down_expected = F.silu(down_fp64 / C)
        padding = activation[rows:, :]
        down_name = "down_wave_splitk_silu"
    assert torch.allclose(down_actual.double(), down_expected, **DOWN_TOLERANCE), f"T={rows}: {down_name} vs FP64"
    assert torch.all(padding == 0), f"T={rows}: down padding must be zero"

    reader.run_up(x)
    torch.cuda.synchronize(x.device)
    assert torch.allclose(reader.output.double(), expected, **OUTPUT_TOLERANCE), f"T={rows}: staged output vs FP64"
    assert torch.allclose(reader.output, combined, **OUTPUT_TOLERANCE), f"T={rows}: staged vs combined output"
    return {
        "rows": rows,
        "down": down_name,
        "partial_elements": reader.partial.numel(),
        "down_max_abs": (down_actual.double() - down_expected).abs().max().item(),
        "output_max_abs": (combined.double() - expected).abs().max().item(),
    }


def check_graph_case(rows):
    x, w_down, w_up = make_case(rows, seed=137)
    reader = CombinedPaddedGRRead(rows, w_down, w_up)
    for _ in range(3):
        reader(x)
    torch.cuda.synchronize(x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = reader(x)
    for replay in range(3):
        x.mul_(0.9).add_(0.03125)
        graph.replay()
        torch.cuda.synchronize(x.device)
        expected = reference_fp64(x, w_down, w_up)
        assert torch.allclose(actual.double(), expected, **OUTPUT_TOLERANCE), f"T={rows}: graph replay {replay + 1}"


def check_input_contract():
    x, w_down, w_up = make_case(1)
    assert CombinedPaddedGRRead(0, w_down, w_up)(x[:0]).shape == (0, H)
    for invalid_rows in (-1, MAX_ROWS + 1):
        try:
            CombinedPaddedGRRead(invalid_rows, w_down, w_up)
        except ValueError as error:
            assert f"0..{MAX_ROWS}" in str(error), str(error)
        else:
            assert False, f"T={invalid_rows}: expected ValueError"

    reader = CombinedPaddedGRRead(1, w_down, w_up)
    for invalid_x, expected_message in (
        (x.float(), "prepared"),
        (x.repeat(2, 1), "prepared"),
        (torch.empty(1, K * 2, dtype=x.dtype, device=x.device)[:, ::2], "contiguous"),
    ):
        try:
            reader(invalid_x)
        except ValueError as error:
            assert expected_message in str(error), str(error)
        else:
            assert False, f"shape={invalid_x.shape}, dtype={invalid_x.dtype}: expected ValueError ({expected_message})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 17, 24, 25, 32])
    parser.add_argument("--graph", action="store_true", help="also verify changed-input graph replay")
    parser.add_argument("--check-contract", action="store_true", help="also check empty and invalid inputs")
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
        print("Empty and invalid input checks passed", flush=True)
