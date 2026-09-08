# -*- coding: utf-8 -*-
"""Corrected common estimator for all six second-order process interactions.

Corrections relative to the original interaction scripts
--------------------------------------------------------
1. Joint contributions and first-order indices use the same corrected total
   variance, normalized model weights, activity mask, and bin definitions.
2. Explicit reaction/flow bin-membership matrices replace fixed-size padded
   arrays. This removes the zero-padding bias in the original climate-flow and
   heat-flow scripts and avoids padding-dependent total variances.
3. The corrected first-order arrays are subtracted from each normalized joint
   contribution; the old biased reaction first-order array is not reused.
4. Every model alternative is averaged exactly once. Nonfinite or zero-variance
   cells return NaN, and raw interaction estimates are saved without clipping.
5. The climate-reaction, climate-flow, reaction-heat, and heat-flow joint terms
   use the same binning approximations as their corresponding first-order terms.
   The reaction-flow joint term retains the 100 paired joint realizations, as in
   the original formulation.
"""

from pathlib import Path
import argparse
import time

import numpy as np

from process_sensitivity_common import (
    DATA_FILE, THRESHOLD, DEFAULT_CHUNK_SIZE,
    W_C, W_H, W_PP, W_PG, W_RXN_BIN, W_FLOW_BIN,
    N_RXN_BINS, N_FLOW_BINS,
)


BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
BASE_DIR.mkdir(parents=True, exist_ok=True)

FIRST_ORDER_FILES = {
    "C": "ts_SI_CS_cc.npy",
    "R": "ts_SI_RT_bin_cc.npy",
    "H": "ts_SI_HT_cc.npy",
    "F": "ts_SI_GF_bin_cc.npy",
}

OUTPUT_FILES = {
    "CR": "ts_Interact_CR_cc.npy",
    "CH": "ts_Interact_CH_cc.npy",
    "CF": "ts_Interact_CF_cc.npy",
    "RH": "ts_Interact_RH_cc.npy",
    "RF": "ts_Interact_RF_cc.npy",
    "HF": "ts_Interact_HF_cc.npy",
}


def _exclusive_interaction(second, mean, variance, first_a, first_b):
    joint = (second - mean ** 2) / variance
    return joint - first_a - first_b


def compute_second_chunk(y, first):
    """Compute the six exclusive second-order indices for one point chunk."""
    y = np.asarray(y)
    reduce_axes = (1, 2, 3, 4, 5)
    inactive = np.all(y <= THRESHOLD, axis=reduce_axes)
    total_variance = np.var(y, axis=reduce_axes, dtype=np.float64)
    valid = (~inactive) & np.isfinite(total_variance) & (total_variance > 0.0)
    for values in first.values():
        valid &= np.isfinite(values)

    results = {
        name: np.full(y.shape[0], np.nan, dtype=np.float64)
        for name in OUTPUT_FILES
    }
    if not np.any(valid):
        return results

    y = y[valid]
    variance = total_variance[valid]
    f = {name: np.asarray(values)[valid] for name, values in first.items()}

    # Average the realization axis for terms that do not require a binning
    # approximation (climate-heat).
    mean_r = np.mean(y, axis=2, dtype=np.float64)  # p,c,h,pp,pg

    # Climate-heat: retain climate and heat; average flow models and parameters.
    m_ch = np.einsum("pchfg,f,g->pch", mean_r, W_PP, W_PG,
                     optimize=True)
    mu_ch = np.einsum("pch,c,h->p", m_ch, W_C, W_H, optimize=True)
    second_ch = np.einsum("pch,c,h->p", m_ch ** 2, W_C, W_H,
                          optimize=True)

    # Reaction-bin conditional means: p,b,c,h,pp,pg.
    cond_rxn = np.einsum("pcrhfg,rb->pbchfg", y, W_RXN_BIN,
                         optimize=True)

    # Climate-reaction: retain climate and reaction bin.
    m_cr = np.einsum("pbchfg,h,f,g->pbc", cond_rxn, W_H, W_PP, W_PG,
                     optimize=True)
    mu_cr = np.mean(np.einsum("pbc,c->pb", m_cr, W_C,
                              optimize=True), axis=1)
    second_cr = np.mean(np.einsum("pbc,c->pb", m_cr ** 2, W_C,
                                  optimize=True), axis=1)

    # Reaction-heat: retain reaction bin and heat model.
    m_rh = np.einsum("pbchfg,c,f,g->pbh", cond_rxn, W_C, W_PP, W_PG,
                     optimize=True)
    mu_rh = np.mean(np.einsum("pbh,h->pb", m_rh, W_H,
                              optimize=True), axis=1)
    second_rh = np.mean(np.einsum("pbh,h->pb", m_rh ** 2, W_H,
                                  optimize=True), axis=1)

    # Flow-bin conditional means: p,pg,pp,b,c,h.
    cond_flow = np.einsum("pcrhfg,gfrb->pgfbch", y, W_FLOW_BIN,
                          optimize=True)
    flow_bin_weight = 1.0 / N_FLOW_BINS

    # Climate-flow: retain climate and all flow-block states; average heat and
    # the paired reaction parameters within each flow bin.
    m_cf = np.einsum("pgfbch,h->pgfbc", cond_flow, W_H, optimize=True)
    mu_cf = np.einsum("pgfbc,g,f,c->p", m_cf, W_PG, W_PP, W_C,
                      optimize=True) * flow_bin_weight
    second_cf = np.einsum("pgfbc,g,f,c->p", m_cf ** 2, W_PG, W_PP, W_C,
                          optimize=True) * flow_bin_weight

    # Heat-flow: retain heat and all flow-block states; average climate and the
    # paired reaction parameters within each flow bin.
    m_hf = np.einsum("pgfbch,c->pgfbh", cond_flow, W_C, optimize=True)
    mu_hf = np.einsum("pgfbh,g,f,h->p", m_hf, W_PG, W_PP, W_H,
                      optimize=True) * flow_bin_weight
    second_hf = np.einsum("pgfbh,g,f,h->p", m_hf ** 2, W_PG, W_PP, W_H,
                          optimize=True) * flow_bin_weight

    # Reaction-flow: the realization axis represents the paired joint state of
    # the reaction and flow parameters. Retain r, permeability model, and
    # geology; average climate and heat.
    m_rf = np.einsum("pcrhfg,c,h->prfg", y, W_C, W_H, optimize=True)
    mu_rf = np.einsum("prfg,f,g->p", m_rf, W_PP, W_PG,
                      optimize=True) / y.shape[2]
    second_rf = np.einsum("prfg,f,g->p", m_rf ** 2, W_PP, W_PG,
                          optimize=True) / y.shape[2]

    computed = {
        "CR": _exclusive_interaction(second_cr, mu_cr, variance, f["C"], f["R"]),
        "CH": _exclusive_interaction(second_ch, mu_ch, variance, f["C"], f["H"]),
        "CF": _exclusive_interaction(second_cf, mu_cf, variance, f["C"], f["F"]),
        "RH": _exclusive_interaction(second_rh, mu_rh, variance, f["R"], f["H"]),
        "RF": _exclusive_interaction(second_rf, mu_rf, variance, f["R"], f["F"]),
        "HF": _exclusive_interaction(second_hf, mu_hf, variance, f["H"], f["F"]),
    }
    for name, values in computed.items():
        results[name][valid] = values
    return results


def _print_diagnostics(name, values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print("{}: no finite estimates".format(name))
        return
    print(
        "{}: n={}, min={:.6g}, q25={:.6g}, median={:.6g}, q75={:.6g}, "
        "max={:.6g}, negative={:.3%}, >1={:.3%}".format(
            name,
            finite.size,
            np.min(finite),
            np.quantile(finite, 0.25),
            np.median(finite),
            np.quantile(finite, 0.75),
            np.max(finite),
            np.mean(finite < 0.0),
            np.mean(finite > 1.0),
        )
    )


def run_all(chunk_size=DEFAULT_CHUNK_SIZE):
    data = np.load(DATA_FILE, mmap_mode="r")
    first = {
        name: np.load(BASE_DIR / filename, mmap_mode="r")
        for name, filename in FIRST_ORDER_FILES.items()
    }
    ntimes, npoints = data.shape[:2]
    outputs = {
        name: np.full((ntimes, npoints), np.nan, dtype=np.float64)
        for name in OUTPUT_FILES
    }
    start_time = time.time()
    for itime in range(ntimes):
        print("Time index {}/{}".format(itime + 1, ntimes))
        for start in range(0, npoints, chunk_size):
            stop = min(start + chunk_size, npoints)
            first_chunk = {
                name: values[itime, start:stop] for name, values in first.items()
            }
            chunk_results = compute_second_chunk(
                data[itime, start:stop], first_chunk
            )
            for name in outputs:
                outputs[name][itime, start:stop] = chunk_results[name]
            if start == 0 or stop == npoints or stop % (chunk_size * 20) == 0:
                print("  points {:,}/{:,}; elapsed {:.1f} min".format(
                    stop, npoints, (time.time() - start_time) / 60.0
                ))

    for name, values in outputs.items():
        output_path = BASE_DIR / OUTPUT_FILES[name]
        np.save(output_path, values)
        _print_diagnostics(name, values)
        print("Saved", output_path)
    return outputs


def run_metric(name, chunk_size=DEFAULT_CHUNK_SIZE):
    if name not in OUTPUT_FILES:
        raise KeyError("Unknown second-order interaction: {}".format(name))
    data = np.load(DATA_FILE, mmap_mode="r")
    first = {
        key: np.load(BASE_DIR / filename, mmap_mode="r")
        for key, filename in FIRST_ORDER_FILES.items()
    }
    ntimes, npoints = data.shape[:2]
    output = np.full((ntimes, npoints), np.nan, dtype=np.float64)
    start_time = time.time()
    for itime in range(ntimes):
        print("{}: time index {}/{}".format(name, itime + 1, ntimes))
        for start in range(0, npoints, chunk_size):
            stop = min(start + chunk_size, npoints)
            first_chunk = {
                key: values[itime, start:stop] for key, values in first.items()
            }
            output[itime, start:stop] = compute_second_chunk(
                data[itime, start:stop], first_chunk
            )[name]
            if start == 0 or stop == npoints or stop % (chunk_size * 20) == 0:
                print("  points {:,}/{:,}; elapsed {:.1f} min".format(
                    stop, npoints, (time.time() - start_time) / 60.0
                ))
    output_path = BASE_DIR / OUTPUT_FILES[name]
    np.save(output_path, output)
    _print_diagnostics(name, output)
    print("Saved", output_path)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=sorted(OUTPUT_FILES))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    if args.metric:
        run_metric(args.metric, args.chunk_size)
    else:
        run_all(args.chunk_size)
