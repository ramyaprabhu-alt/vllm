#!/usr/bin/env python3
"""Compare two bubble sweep JSON files side by side."""
import json
import sys

def load(path):
    with open(path) as f:
        rows = json.load(f)
    return {(r["t_in"], r["bs"]): r for r in rows}

def fmt(v, decimals=1):
    return f"{v:.{decimals}f}" if v is not None else "  —  "

def main():
    path_a = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bubble_sweep_no_eplb.json"
    path_b = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bubble_sweep_eplb.json"
    label_a = sys.argv[3] if len(sys.argv) > 3 else "no-EPLB"
    label_b = sys.argv[4] if len(sys.argv) > 4 else "EPLB"

    a = load(path_a)
    b = load(path_b)
    all_keys = sorted(set(a) | set(b))

    # Header
    W = 8
    print(f"\n{'':>12}  "
          f"{'─── ' + label_a + ' ───':^52}  "
          f"{'─── ' + label_b + ' ───':^52}  "
          f"{'── EPLB effect ──':^22}")
    print(f"{'T_in×BS':>12}  "
          f"{'tok':>5} {'r0_ffn':>7} {'r1_ffn':>7} {'imbal%':>7} {'ar_mean':>7} {'ar%':>5}  "
          f"{'tok':>5} {'r0_ffn':>7} {'r1_ffn':>7} {'imbal%':>7} {'ar_mean':>7} {'ar%':>5}  "
          f"{'imbal_Δ':>8} {'ffn_Δ%':>7} {'lat_Δms':>8}")
    print("─" * 130)

    for key in all_keys:
        t_in, bs = key
        ra = a.get(key)
        rb = b.get(key)

        def fields(r):
            if r is None:
                return "  —  ", "  —  ", "  —  ", "  —  ", "  —  ", "  —  "
            tok   = f"{r['actual_tok']:>5.0f}"
            r0f   = f"{r['r0_ffn_sum']:>7.1f}"
            r1f   = f"{r['r1_ffn_sum']:>7.1f}"
            imbal = f"{r['imbalance_pct']:>7.1f}"
            ar    = f"{r['r0_ar_mean']:>7.2f}"
            arp   = f"{r['ar_frac_pct']:>5.1f}"
            return tok, r0f, r1f, imbal, ar, arp

        fa = fields(ra)
        fb = fields(rb)

        # Delta columns
        if ra and rb:
            imbal_delta = rb["imbalance_pct"] - ra["imbalance_pct"]
            ffn_delta   = (rb["r0_ffn_sum"] - ra["r0_ffn_sum"]) / ra["r0_ffn_sum"] * 100
            lat_delta   = rb["lat_ms"] - ra["lat_ms"]
            delta_str   = (f"{imbal_delta:>+8.1f} {ffn_delta:>+7.1f} {lat_delta:>+8.0f}")
        else:
            delta_str = f"{'':>8} {'':>7} {'':>8}"

        print(f"{t_in:>6}×{bs:<4}  "
              f"{fa[0]} {fa[1]} {fa[2]} {fa[3]} {fa[4]} {fa[5]}  "
              f"{fb[0]} {fb[1]} {fb[2]} {fb[3]} {fb[4]} {fb[5]}  "
              f"{delta_str}")

    print()
    print("Columns: tok=actual tokens processed, r0/r1_ffn=FFN sum over 48 layers (ms),")
    print("         imbal%=|r0-r1|/max×100, ar_mean=mean allreduce/layer (ms), ar%=allreduce fraction")
    print("Deltas:  EPLB minus no-EPLB  (+ = EPLB is higher/slower)")

if __name__ == "__main__":
    main()
