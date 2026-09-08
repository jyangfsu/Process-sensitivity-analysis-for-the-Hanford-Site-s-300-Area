# -*- coding: utf-8 -*-
"""Corrected estimator for four third-order and one fourth-order interactions.

Corrections relative to the original third-order scripts
--------------------------------------------------------
1. All joint contributions use the original ensemble variance, normalized model
   weights, explicit bin-membership means, and the common activity mask.
2. The climate-heat-flow calculation no longer uses zero padding and no longer
   overwrites its six-climate-model average with two heat-model weights.
3. Exclusive third-order terms subtract the corrected first- and second-order
   components for the corresponding subset.
4. The fourth-order term is recomputed by variance closure from all corrected
   lower-order components. Raw estimates are saved without clipping.
5. Nonfinite or zero-variance cells return NaN. The 100 paired realization
   states are retained when both reaction and flow belong to the joint subset.
"""

from pathlib import Path
import argparse
import time

import numpy as np

from process_sensitivity_common import (
    DATA_FILE, THRESHOLD, DEFAULT_CHUNK_SIZE,
    W_C, W_H, W_PP, W_PG, W_RXN_BIN, W_FLOW_BIN, N_FLOW_BINS,
)


BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
BASE_DIR.mkdir(parents=True, exist_ok=True)

FIRST_ORDER_FILES = {
    "C": "ts_SI_CS_cc.npy",
    "R": "ts_SI_RT_bin_cc.npy",
    "H": "ts_SI_HT_cc.npy",
    "F": "ts_SI_GF_bin_cc.npy",
}

SECOND_ORDER_FILES = {
    "CR": "ts_Interact_CR_cc.npy",
    "CH": "ts_Interact_CH_cc.npy",
    "CF": "ts_Interact_CF_cc.npy",
    "RH": "ts_Interact_RH_cc.npy",
    "RF": "ts_Interact_RF_cc.npy",
    "HF": "ts_Interact_HF_cc.npy",
}

THIRD_ORDER_FILES = {
    "CRH": "ts_Interact_CRH_cc.npy",
    "CRF": "ts_Interact_CRF_cc.npy",
    "CFH": "ts_Interact_CHF_cc.npy",
    "RHF": "ts_Interact_RHF_cc.npy",
}

FOURTH_ORDER_FILE = "ts_Interact_CFHR_cc.npy"


def _exclusive_third(second_moment, mean, variance, lower_terms):
    joint = (second_moment - mean ** 2) / variance
    exclusive = joint.copy()
    for term in lower_terms:
        exclusive -= term
    return exclusive


def compute_third_chunk(y, first, second):
    """Compute all four exclusive third-order indices for a point chunk."""
    y = np.asarray(y)
    reduce_axes = (1, 2, 3, 4, 5)
    inactive = np.all(y <= THRESHOLD, axis=reduce_axes)
    total_variance = np.var(y, axis=reduce_axes, dtype=np.float64)
    valid = (~inactive) & np.isfinite(total_variance) & (total_variance > 0.0)
    for values in first.values():
        valid &= np.isfinite(values)
    for values in second.values():
        valid &= np.isfinite(values)

    results = {
        name: np.full(y.shape[0], np.nan, dtype=np.float64)
        for name in THIRD_ORDER_FILES
    }
    if not np.any(valid):
        return results

    y = y[valid]
    variance = total_variance[valid]
    f = {name: np.asarray(values)[valid] for name, values in first.items()}
    s = {name: np.asarray(values)[valid] for name, values in second.items()}

    # Reaction-bin conditional means: p,b,c,h,pp,pg. Averaging pp and pg leaves
    # the climate-reaction-heat joint state.
    cond_rxn = np.einsum("pcrhfg,rb->pbchfg", y, W_RXN_BIN,
                         optimize=True)
    m_crh = np.einsum("pbchfg,f,g->pbch", cond_rxn, W_PP, W_PG,
                      optimize=True)
    mu_crh = np.mean(np.einsum("pbch,c,h->pb", m_crh, W_C, W_H,
                               optimize=True), axis=1)
    second_crh = np.mean(np.einsum("pbch,c,h->pb", m_crh ** 2, W_C, W_H,
                                   optimize=True), axis=1)

    # Climate-reaction-flow retains the climate model and the 100 paired joint
    # reaction-flow realization states; only heat is averaged out.
    m_crf = np.einsum("pcrhfg,h->pcrfg", y, W_H, optimize=True)
    mu_crf = np.einsum("pcrfg,c,f,g->p", m_crf, W_C, W_PP, W_PG,
                       optimize=True) / y.shape[2]
    second_crf = np.einsum("pcrfg,c,f,g->p", m_crf ** 2, W_C, W_PP, W_PG,
                           optimize=True) / y.shape[2]

    # Flow-bin conditional means: p,pg,pp,b,c,h. These retain climate, heat,
    # and the complete flow block while averaging paired reaction parameters.
    cond_flow = np.einsum("pcrhfg,gfrb->pgfbch", y, W_FLOW_BIN,
                          optimize=True)
    flow_bin_weight = 1.0 / N_FLOW_BINS
    mu_cfh = np.einsum("pgfbch,g,f,c,h->p", cond_flow, W_PG, W_PP, W_C, W_H,
                       optimize=True) * flow_bin_weight
    second_cfh = np.einsum(
        "pgfbch,g,f,c,h->p", cond_flow ** 2, W_PG, W_PP, W_C, W_H,
        optimize=True
    ) * flow_bin_weight

    # Reaction-heat-flow retains heat and all paired reaction-flow states;
    # climate is averaged out.
    m_rhf = np.einsum("pcrhfg,c->prhfg", y, W_C, optimize=True)
    mu_rhf = np.einsum("prhfg,h,f,g->p", m_rhf, W_H, W_PP, W_PG,
                       optimize=True) / y.shape[2]
    second_rhf = np.einsum("prhfg,h,f,g->p", m_rhf ** 2, W_H, W_PP, W_PG,
                           optimize=True) / y.shape[2]

    computed = {
        "CRH": _exclusive_third(
            second_crh, mu_crh, variance,
            (f["C"], f["R"], f["H"], s["CR"], s["CH"], s["RH"]),
        ),
        "CRF": _exclusive_third(
            second_crf, mu_crf, variance,
            (f["C"], f["R"], f["F"], s["CR"], s["CF"], s["RF"]),
        ),
        "CFH": _exclusive_third(
            second_cfh, mu_cfh, variance,
            (f["C"], f["F"], f["H"], s["CF"], s["CH"], s["HF"]),
        ),
        "RHF": _exclusive_third(
            second_rhf, mu_rhf, variance,
            (f["R"], f["H"], f["F"], s["RH"], s["RF"], s["HF"]),
        ),
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
            name, finite.size, np.min(finite), np.quantile(finite, 0.25),
            np.median(finite), np.quantile(finite, 0.75), np.max(finite),
            np.mean(finite < 0.0), np.mean(finite > 1.0),
        )
    )


def _load_lower_orders():
    first = {
        name: np.load(BASE_DIR / filename, mmap_mode="r")
        for name, filename in FIRST_ORDER_FILES.items()
    }
    second = {
        name: np.load(BASE_DIR / filename, mmap_mode="r")
        for name, filename in SECOND_ORDER_FILES.items()
    }
    return first, second


def _save_fourth(first, second, third):
    valid = np.ones_like(next(iter(first.values())), dtype=bool)
    for collection in (first, second, third):
        for values in collection.values():
            valid &= np.isfinite(values)
    fourth = np.full(valid.shape, np.nan, dtype=np.float64)
    lower_sum = np.zeros(valid.shape, dtype=np.float64)
    for collection in (first, second, third):
        for values in collection.values():
            lower_sum[valid] += values[valid]
    fourth[valid] = 1.0 - lower_sum[valid]
    output_path = BASE_DIR / FOURTH_ORDER_FILE
    np.save(output_path, fourth)
    _print_diagnostics("CFHR", fourth)
    print("Saved", output_path)
    closure = lower_sum[valid] + fourth[valid]
    print("Closure max absolute error:", float(np.max(np.abs(closure - 1.0))))
    return fourth


def run_all(chunk_size=DEFAULT_CHUNK_SIZE):
    data = np.load(DATA_FILE, mmap_mode="r")
    first, second = _load_lower_orders()
    ntimes, npoints = data.shape[:2]
    outputs = {
        name: np.full((ntimes, npoints), np.nan, dtype=np.float64)
        for name in THIRD_ORDER_FILES
    }
    start_time = time.time()
    for itime in range(ntimes):
        print("Time index {}/{}".format(itime + 1, ntimes))
        for start in range(0, npoints, chunk_size):
            stop = min(start + chunk_size, npoints)
            first_chunk = {
                name: values[itime, start:stop] for name, values in first.items()
            }
            second_chunk = {
                name: values[itime, start:stop] for name, values in second.items()
            }
            chunk_results = compute_third_chunk(
                data[itime, start:stop], first_chunk, second_chunk
            )
            for name in outputs:
                outputs[name][itime, start:stop] = chunk_results[name]
            if start == 0 or stop == npoints or stop % (chunk_size * 20) == 0:
                print("  points {:,}/{:,}; elapsed {:.1f} min".format(
                    stop, npoints, (time.time() - start_time) / 60.0
                ))

    for name, values in outputs.items():
        output_path = BASE_DIR / THIRD_ORDER_FILES[name]
        np.save(output_path, values)
        _print_diagnostics(name, values)
        print("Saved", output_path)
    _save_fourth(first, second, outputs)
    return outputs


def run_metric(name, chunk_size=DEFAULT_CHUNK_SIZE):
    if name not in THIRD_ORDER_FILES:
        raise KeyError("Unknown third-order interaction: {}".format(name))
    data = np.load(DATA_FILE, mmap_mode="r")
    first, second = _load_lower_orders()
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
            second_chunk = {
                key: values[itime, start:stop] for key, values in second.items()
            }
            output[itime, start:stop] = compute_third_chunk(
                data[itime, start:stop], first_chunk, second_chunk
            )[name]
            if start == 0 or stop == npoints or stop % (chunk_size * 20) == 0:
                print("  points {:,}/{:,}; elapsed {:.1f} min".format(
                    stop, npoints, (time.time() - start_time) / 60.0
                ))
    output_path = BASE_DIR / THIRD_ORDER_FILES[name]
    np.save(output_path, output)
    _print_diagnostics(name, output)
    print("Saved", output_path)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=sorted(THIRD_ORDER_FILES))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    if args.metric:
        run_metric(args.metric, args.chunk_size)
    else:
        run_all(args.chunk_size)
