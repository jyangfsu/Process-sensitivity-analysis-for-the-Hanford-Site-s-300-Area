# -*- coding: utf-8 -*-
"""Corrected common estimator for first-order and total-effect process indices.

Corrections relative to the original PSI/PST scripts
----------------------------------------------------
1. The total output variance is always computed from the original 12,000-member
   ensemble at each grid cell, never from a padded bin-storage array.
2. Conditional bin means are evaluated with explicit membership matrices; no
   zero padding or all-NaN ``nansum`` operation can enter an expectation.
3. Every alternative process model is included exactly once with a normalized
   model weight, including both heat models in the reaction total effect.
4. Reaction and flow bins are treated as equal-probability bins, consistent with
   the quantile/rank partitions used by the original implementation. Samples
   within each occupied bin receive equal conditional weights.
5. Nonfinite or zero-variance cells return NaN. Raw estimates are saved without
   clipping so that negative or greater-than-one estimates remain diagnosable.
6. The original activity-mask implementation is retained for comparability:
   a cell is excluded only when all 12,000 outputs are <= 1e-7.

The input layout is
    [time, point, climate, realization, heat, permeability model, geology].
"""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "common"))

from constvars import (
    Mps, Mph, Mpp, Mpg, PMps, PMph, PMpp, PMpg,
    k1, k2, k3, hanford_homo_perm, alluvium_homo_perm,
    hanford_hete_perm_rank, alluvium_hete_perm_rank,
)


BASE_DIR = PROJECT_ROOT / "data" / "processed"
BASE_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "doc_consum.npy"
THRESHOLD = 1.0e-7
DEFAULT_CHUNK_SIZE = 512

OUTPUT_FILES = {
    "PS_C": "ts_SI_CS_cc.npy",
    "PS_R": "ts_SI_RT_bin_cc.npy",
    "PS_H": "ts_SI_HT_cc.npy",
    "PS_F": "ts_SI_GF_bin_cc.npy",
    "PST_C": "ts_ST_CS_cc.npy",
    "PST_R": "ts_ST_RT_bin_cc.npy",
    "PST_H": "ts_ST_HT_cc.npy",
    "PST_F": "ts_ST_GF_bin_cc.npy",
}


def _normalized(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Process-model weights must have a positive finite sum.")
    return weights / total


W_C = _normalized(PMps)
W_H = _normalized(PMph)
W_PP = _normalized(PMpp)
W_PG = _normalized(PMpg)


def _membership_matrix(bin_numbers, expected_bins, label):
    """Return sample-within-bin averaging weights of shape (sample, bin)."""
    bin_numbers = np.asarray(bin_numbers, dtype=np.int64)
    unique_numbers = np.unique(bin_numbers)
    if unique_numbers.size != expected_bins:
        raise ValueError(
            "{} has {} occupied bins; {} were expected.".format(
                label, unique_numbers.size, expected_bins
            )
        )
    matrix = np.zeros((bin_numbers.size, expected_bins), dtype=np.float64)
    occupancies = np.zeros(expected_bins, dtype=np.int64)
    # scipy.stats uses padded linear bin numbers when expand_binnumbers=False;
    # map the sorted occupied labels to compact columns rather than assuming
    # that the returned labels are the consecutive integers 1...H.
    for ibin, number in enumerate(unique_numbers):
        members = np.flatnonzero(bin_numbers == number)
        occupancies[ibin] = members.size
        matrix[members, ibin] = 1.0 / members.size
    return matrix, occupancies


def _reaction_membership():
    nbins = 3
    edges = [
        stats.uniform.ppf(np.linspace(0, 1, nbins + 1), loc=0, scale=28.26 * 2),
        stats.uniform.ppf(np.linspace(0, 1, nbins + 1), loc=0, scale=23.28 * 2),
        stats.uniform.ppf(np.linspace(0, 1, nbins + 1), loc=0, scale=84.78 * 2),
    ]
    _, _, numbers = stats.binned_statistic_dd(
        np.vstack([k1, k2, k3]).T,
        np.zeros(len(k1)),
        statistic="count",
        bins=edges,
        expand_binnumbers=False,
    )
    return _membership_matrix(numbers, nbins ** 3, "Reaction")


def _flow_membership():
    nbins = 4
    matrix = np.zeros((Mpg, Mpp, 100, nbins ** 2), dtype=np.float64)
    occupancies = np.zeros((Mpg, Mpp, nbins ** 2), dtype=np.int64)
    hete_edges = stats.uniform.ppf(
        np.linspace(0, 1, nbins + 1), loc=0, scale=3000
    )
    homo_h_edges = [1.00e-11, 1.90e-9, 7.27e-9, 2.76e-8, 1.00e-6]
    homo_a_edges = [1.00e-13, 7.53e-13, 1.06e-12, 1.48e-12, 1.00e-11]

    for ipg in range(Mpg):
        for ipp in range(Mpp):
            if ipp == 0:
                h_values = hanford_hete_perm_rank[:, ipg]
                a_values = alluvium_hete_perm_rank[:, ipg]
                edges = [hete_edges, hete_edges]
            else:
                h_values = hanford_homo_perm
                a_values = alluvium_homo_perm
                edges = [homo_h_edges, homo_a_edges]
            _, _, _, numbers = stats.binned_statistic_2d(
                h_values,
                a_values,
                np.zeros(100),
                statistic="count",
                bins=edges,
                expand_binnumbers=False,
            )
            weights, counts = _membership_matrix(
                numbers, nbins ** 2, "Flow ({}, {})".format(ipg, ipp)
            )
            matrix[ipg, ipp] = weights
            occupancies[ipg, ipp] = counts
    return matrix, occupancies


W_RXN_BIN, RXN_OCCUPANCIES = _reaction_membership()
W_FLOW_BIN, FLOW_OCCUPANCIES = _flow_membership()
N_RXN_BINS = W_RXN_BIN.shape[1]
N_FLOW_BINS = W_FLOW_BIN.shape[-1]


def compute_chunk(y):
    """Compute all eight indices for a point chunk.

    Parameters
    ----------
    y : ndarray
        Shape (point, climate, realization, heat, permeability model, geology).
    """
    y = np.asarray(y)
    reduce_axes = (1, 2, 3, 4, 5)
    inactive = np.all(y <= THRESHOLD, axis=reduce_axes)
    total_variance = np.var(y, axis=reduce_axes, dtype=np.float64)
    valid = (~inactive) & np.isfinite(total_variance) & (total_variance > 0.0)

    results = {
        name: np.full(y.shape[0], np.nan, dtype=np.float64)
        for name in OUTPUT_FILES
    }
    if not np.any(valid):
        return results

    y = y[valid]
    variance = total_variance[valid]

    # First-order climate and heat indices.
    mean_r = np.mean(y, axis=2, dtype=np.float64)  # p,c,h,pp,pg
    m_c = np.einsum("pchfg,h,f,g->pc", mean_r, W_H, W_PP, W_PG,
                    optimize=True)
    mu_c = np.einsum("pc,c->p", m_c, W_C, optimize=True)
    c_c = np.einsum("pc,c->p", m_c ** 2, W_C, optimize=True) - mu_c ** 2

    m_h = np.einsum("pchfg,c,f,g->ph", mean_r, W_C, W_PP, W_PG,
                    optimize=True)
    mu_h = np.einsum("ph,h->p", m_h, W_H, optimize=True)
    c_h = np.einsum("ph,h->p", m_h ** 2, W_H, optimize=True) - mu_h ** 2

    # Reaction-bin conditional means: p,b,c,h,pp,pg.
    cond_rxn = np.einsum("pcrhfg,rb->pbchfg", y, W_RXN_BIN,
                         optimize=True)
    m_r = np.einsum("pbchfg,c,h,f,g->pb", cond_rxn, W_C, W_H, W_PP,
                    W_PG, optimize=True)
    mu_r = np.mean(m_r, axis=1)
    c_r = np.mean(m_r ** 2, axis=1) - mu_r ** 2

    # The same reaction-bin means provide the complement of flow for PST_F.
    m_not_f = np.einsum("pbchfg,f,g->pbch", cond_rxn, W_PP, W_PG,
                        optimize=True)
    mu_not_f = np.mean(
        np.einsum("pbch,c,h->pb", m_not_f, W_C, W_H, optimize=True), axis=1
    )
    second_not_f = np.mean(
        np.einsum("pbch,c,h->pb", m_not_f ** 2, W_C, W_H,
                  optimize=True), axis=1
    )
    c_not_f = second_not_f - mu_not_f ** 2

    # Flow-bin conditional means: p,pg,pp,b,c,h.
    cond_flow = np.einsum("pcrhfg,gfrb->pgfbch", y, W_FLOW_BIN,
                          optimize=True)
    m_f = np.einsum("pgfbch,c,h->pgfb", cond_flow, W_C, W_H,
                    optimize=True)
    bin_weight_f = 1.0 / N_FLOW_BINS
    mu_f = np.einsum("pgfb,g,f->p", m_f, W_PG, W_PP,
                     optimize=True) * bin_weight_f
    second_f = np.einsum("pgfb,g,f->p", m_f ** 2, W_PG, W_PP,
                         optimize=True) * bin_weight_f
    c_f = second_f - mu_f ** 2

    # The flow-bin conditional means provide the complement of reaction.
    mu_not_r = np.einsum("pgfbch,g,f,c,h->p", cond_flow, W_PG, W_PP,
                         W_C, W_H, optimize=True) * bin_weight_f
    second_not_r = np.einsum(
        "pgfbch,g,f,c,h->p", cond_flow ** 2, W_PG, W_PP, W_C, W_H,
        optimize=True
    ) * bin_weight_f
    c_not_r = second_not_r - mu_not_r ** 2

    # Complement of climate: average climate while retaining r,h,pp,pg.
    m_not_c = np.einsum("pcrhfg,c->prhfg", y, W_C, optimize=True)
    mu_not_c = np.einsum("prhfg,h,f,g->p", m_not_c, W_H, W_PP, W_PG,
                         optimize=True) / y.shape[2]
    second_not_c = np.einsum(
        "prhfg,h,f,g->p", m_not_c ** 2, W_H, W_PP, W_PG,
        optimize=True
    ) / y.shape[2]
    c_not_c = second_not_c - mu_not_c ** 2

    # Complement of heat: average heat while retaining c,r,pp,pg.
    m_not_h = np.einsum("pcrhfg,h->pcrfg", y, W_H, optimize=True)
    mu_not_h = np.einsum("pcrfg,c,f,g->p", m_not_h, W_C, W_PP, W_PG,
                         optimize=True) / y.shape[2]
    second_not_h = np.einsum(
        "pcrfg,c,f,g->p", m_not_h ** 2, W_C, W_PP, W_PG,
        optimize=True
    ) / y.shape[2]
    c_not_h = second_not_h - mu_not_h ** 2

    computed = {
        "PS_C": c_c / variance,
        "PS_R": c_r / variance,
        "PS_H": c_h / variance,
        "PS_F": c_f / variance,
        "PST_C": 1.0 - c_not_c / variance,
        "PST_R": 1.0 - c_not_r / variance,
        "PST_H": 1.0 - c_not_h / variance,
        "PST_F": 1.0 - c_not_f / variance,
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
        "{}: n={}, min={:.6g}, median={:.6g}, max={:.6g}, "
        "negative={:.3%}, >1={:.3%}".format(
            name,
            finite.size,
            np.min(finite),
            np.median(finite),
            np.max(finite),
            np.mean(finite < 0.0),
            np.mean(finite > 1.0),
        )
    )


def run_all(chunk_size=DEFAULT_CHUNK_SIZE):
    data = np.load(DATA_FILE, mmap_mode="r")
    if data.ndim != 7:
        raise ValueError("Unexpected doc_consum shape: {}".format(data.shape))
    ntimes, npoints = data.shape[:2]
    outputs = {
        name: np.full((ntimes, npoints), np.nan, dtype=np.float64)
        for name in OUTPUT_FILES
    }

    print("Input:", DATA_FILE)
    print("Shape:", data.shape, "dtype:", data.dtype)
    print("Reaction-bin occupancies:", RXN_OCCUPANCIES.tolist())
    print(
        "Flow-bin occupancy range:",
        int(FLOW_OCCUPANCIES.min()),
        "to",
        int(FLOW_OCCUPANCIES.max()),
    )
    start_time = time.time()
    for itime in range(ntimes):
        print("Time index {}/{}".format(itime + 1, ntimes))
        for start in range(0, npoints, chunk_size):
            stop = min(start + chunk_size, npoints)
            chunk_results = compute_chunk(data[itime, start:stop])
            for name in outputs:
                outputs[name][itime, start:stop] = chunk_results[name]
            if start == 0 or stop == npoints or stop % (chunk_size * 20) == 0:
                elapsed = time.time() - start_time
                print("  points {:,}/{:,}; elapsed {:.1f} min".format(
                    stop, npoints, elapsed / 60.0
                ))

    for name, values in outputs.items():
        output_path = BASE_DIR / OUTPUT_FILES[name]
        np.save(output_path, values)
        _print_diagnostics(name, values)
        print("Saved", output_path)
    return outputs


def run_metric(name, chunk_size=DEFAULT_CHUNK_SIZE):
    if name not in OUTPUT_FILES:
        raise KeyError("Unknown metric: {}".format(name))
    data = np.load(DATA_FILE, mmap_mode="r")
    ntimes, npoints = data.shape[:2]
    output = np.full((ntimes, npoints), np.nan, dtype=np.float64)
    start_time = time.time()
    for itime in range(ntimes):
        print("{}: time index {}/{}".format(name, itime + 1, ntimes))
        for start in range(0, npoints, chunk_size):
            stop = min(start + chunk_size, npoints)
            output[itime, start:stop] = compute_chunk(data[itime, start:stop])[name]
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
