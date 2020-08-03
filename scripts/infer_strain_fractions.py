#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from lib.pileup import (
    list_samples,
    load_pileup_data,
    sample_sites,
    get_pileup_dims,
    filter_samples,
    convert_to_major_minor_allele,
)
from lib.util import info
from lib.genotype_mixture import (
    build_biallelic_model3,
    build_biallelic_model6,
    pileup_to_model_input,
    ambiguity_index,
    stick_breaking,
)

import numpy as np
import pymc3 as pm
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import sys
import warnings
import argparse


MODELS = {
    "betabinom_observation": build_biallelic_model3,
    "weighted_penalty": build_biallelic_model6,
}


def find_MAP_loop(model, start, verbose=2, **kwargs):
    i = 1
    while True:
        mapest, optim = pm.find_MAP(
            model=model,
            return_raw=True,
            maxeval=10000,
            start=start,
            progressbar=(verbose >= 2),
            **kwargs,
        )
        if verbose >= 2:
            print()  # Add space so that progress bar is not lost.
        # optim returns as None if it reaches maxeval or it gets a SIGKILL.
        if optim is not None:
            logp = model.logp(mapest)
            # TODO: Better error handling.
            assert optim.success and np.isfinite(
                logp
            ), f"Optimization failed:\n{optim}"
            break

        start = mapest
        i += 1
        if verbose > 1:
            print(
                f"MAP estimate has not yet converged. "
                f"Starting round {i} of gradient descent."
            )

    return mapest


def find_MAP_loop_retry(model, multitrace, **kwargs):
    logp = model.logp
    to_try = sorted(
        [multitrace.point(-1, chain=i) for i in multitrace.chains],
        reverse=True,
        key=logp,
    )

    _first = True
    for start in to_try:
        # Inform user of retries.
        if not _first:
            info("Attempting function optimization with next chain terminus.")
        else:
            _first = False

        try:
            mapest = find_MAP_loop(model, start=start, **kwargs)
        # TODO: Update this once error handling in find_MAP_loop gets better.
        except AssertionError as err:
            info(err)
        else:
            return mapest
    else:
        # TODO: Better error handling.
        raise RuntimeError(
            "No chain termini produced a valid MAP estimate.  "
            "Try adjusting the fitting parameters."
        )


def list_strains(n, pad=3):
    frmt = "s{:0" + str(pad) + "d}"
    out = [frmt.format(i) for i in range(1, n + 1)]
    assert len(out) == n
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input
    p.add_argument(
        "pileup",
        help="""
Pileup in sparse format with columns: [segment_id, position, sample_id, base,
tally].  The file should have a header row, but the names themselves are
ignored.
                        """,
    )

    # Shape of the model
    p.add_argument("--nstrains", metavar="FLOAT", type=int, default=30)
    p.add_argument(
        "--npos",
        metavar="INT",
        default=100,
        type=int,
        help=("Number of variable positions to sample for model " "fitting."),
    )
    p.add_argument(
        "--model",
        default="betabinom_observation",
        help="Which model to fit.",
        choices=MODELS.keys(),
    )

    # Regularization
    p.add_argument(
        "--heterogeneity-penalty",
        metavar="FLOAT",
        default=0,
        type=float,
        help=("TODO"),
    )
    p.add_argument(
        "--diversity-penalty",
        metavar="FLOAT",
        default=0,
        type=float,
        help=("TODO"),
    )
    p.add_argument(
        "--ambiguity-penalty",
        metavar="FLOAT",
        default=0,
        type=float,
        help=("TODO"),
    )
    p.add_argument("--error-prior", metavar="FLOAT", default=100, type=float)
    p.add_argument(
        "--concentration",
        metavar="FLOAT",
        default=100,
        type=float,
        help=('Only affects "--model betabinom_observation"'),
    )
    p.add_argument(
        "--fingerprint-ambiguity-penalty",
        metavar="FLOAT",
        default=0,
        type=float,
        help=("TODO"),
    )
    p.add_argument("--derep-thresh", default=0, type=float, help=("TODO"))

    # Reporting thresholds
    p.add_argument(
        "--min-max-frac",
        metavar="FLOAT",
        default=1e-3,
        type=float,
        help=(
            "Only report genotypes with at least one sample with "
            "fraction greater than this threshold."
        ),
    )
    p.add_argument(
        "--trunc-frac",
        metavar="FLOAT",
        default=1e-5,
        type=float,
        help=("Strain fractions less than this value are dropped."),
    )

    # Stochastic search
    p.add_argument("--random-seed", default=0, type=int, help=("TODO"))
    p.add_argument(
        "--jitter-scale",
        default=1,
        type=float,
        help=(
            "Scale of the jitters introduced to initialize each "
            "stochastic search chain. "
            "With large values, chains may fail due to "
            "hitting numerical boundary conditions."
        ),
    )
    p.add_argument("--nsteps0", default=0, type=int, help=("TODO"))
    p.add_argument("--nsteps", default=50, type=int, help=("TODO"))
    p.add_argument("--nrestarts", default=5, type=int, help=("TODO"))
    p.add_argument("--nchains", default=10, type=int, help=("TODO"))
    p.add_argument("--nprocs", default=1, type=int, help=("TODO"))

    # Output paths
    p.add_argument(
        "--frac-out",
        metavar="PATH",
        help=("Path for genotype fraction output."),
    )
    p.add_argument(
        "--err-out",
        metavar="PATH",
        help=("Path for sample error rate output."),
    )
    p.add_argument(
        "--logp-out",
        metavar="PATH",
        help=("Path for search trace model log-prob output."),
    )

    args = p.parse_args(argv)

    # Defaults
    # if args.frac_out is None:
    #     args.frac_out = f'{args.pileup}.genotype.frac.tsv'
    # if args.err_out is None:
    #     args.err_out = f'{args.pileup}.genotype.sample_error_rate.tsv'
    # # if args.stats_out is None:
    # #     args.stats_out = f'{args.pileup}.genotype.stats.tsv'
    # if args.logp_out is None:
    #     args.logp_out = f'{args.pileup}.genotype.search_logp.tsv'
    if args.nsteps0 == 0:
        args.nsteps0 = args.nsteps

    # Validation
    assert args.nstrains > 1
    assert args.npos > 0
    assert (args.min_max_frac >= 0) and (args.min_max_frac < 1)
    assert (args.trunc_frac >= 0) and (args.trunc_frac < args.min_max_frac)
    assert args.nchains > 0
    assert args.nprocs > 0
    assert args.nsteps > 0
    assert args.nsteps0 > 0
    assert args.nrestarts > 0

    return args


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module="numpy.core.fromnumeric",
        lineno=3335,
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pymc3.sampling", lineno=436
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="numpy.lib.arraysetops",
        lineno=568,
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="theano.tensor.subtensor",
        lineno=2197,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pymc3.tuning.starting",
        lineno=61,
    )
    # warnings.filterwarnings('ignore', category=UserWarning,
    #                         module='pymc3.sampling', lineno=496)
    # Ignore well-intentioned warning from PyMC3:
    #  "The number of samples is too small to check convergence reliably."
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pymc3.sampling", lineno=566
    )

    args = parse_args(sys.argv[1:])
    info(args)

    np.random.seed(args.random_seed)

    info(f"Beginning preparation phase.")
    info(f"Loading data from {args.pileup}")
    pileup = load_pileup_data(args.pileup)
    info(f"Filtering samples.")
    pileup = filter_samples(pileup, min_median_coverage=1)
    if pileup.empty:
        warnings.warn(
            "Pileup is empty after filtering. " "Writing empty output files."
        )
        if args.frac_out:
            with open(args.frac_out, "w"):
                pass
        if args.err_out:
            with open(args.err_out, "w"):
                pass
        if args.logp_out:
            with open(args.logp_out, "w"):
                pass
        sys.exit(0)

    assert not pileup.empty, "Pileup is empty after filtering."
    pileup, variants = convert_to_major_minor_allele(pileup)
    g, n, a = get_pileup_dims(pileup)
    assert (g > 0) and (n > 0) and (a == 2), "Pileup is empty after filtering."
    info(f"(num_positions={g}, num_samples={n}")

    info(f"Median sequence depth at polymorphic sites (histogram of samples):")
    per_sample_median_depth = (
        pileup.groupby(level="sample_id", axis="columns").sum().median()
    )
    bin_width = 5
    hist_bins = np.arange(
        0,
        round(per_sample_median_depth.max()) + bin_width,
        bin_width,
        dtype=int,
    )
    depth_hist = np.histogram(per_sample_median_depth, bins=hist_bins)
    info("median (bin)\tsamples (count)")
    for _count, bin_start in zip(*depth_hist):
        if _count == 0:
            continue
        bin_end = bin_start + bin_width
        info(f"[{bin_start},{bin_end})\t\t{_count}")

    info(f"Fitting model to nstrains={args.nstrains} latent strains.")

    g_sample = min(args.npos, g)
    info(f"Sampling {g_sample} of {g} positions from input data.")
    pileup_sample = sample_sites(pileup, g_sample, random_state=0)
    y_sample = pileup_to_model_input(pileup_sample)

    info("Fitting model.")
    model = MODELS[args.model](g_sample, n, args.nstrains)
    model.observed.set_value(y_sample.reshape((-1, a)))
    model.alpha.set_value(args.concentration)
    # Set hyperparams
    model.gamma_hyper.set_value(args.ambiguity_penalty)
    model.pi_hyper.set_value(args.heterogeneity_penalty)
    model.rho_hyper.set_value(args.diversity_penalty)
    model.epsilon_hyper.set_value(args.error_prior)

    start = model.bijection.map(model.test_point)
    start = [
        start + (args.jitter_scale * (2 * np.random.rand(*start.shape) - 1))
        for _ in range(args.nchains)
    ]
    start = [model.bijection.rmap(point) for point in start]
    search = None
    init = "adapt_diag"
    tune = args.nsteps0
    for i in range(1, args.nrestarts + 1):
        info(f"Stochastic search restart {i} of {args.nrestarts}.")
        with model:
            search = pm.sample(
                start=start,
                chains=args.nchains,
                cores=args.nprocs,
                trace=search,
                init=init,
                tune=tune,
                draws=0,
                discard_tuned_samples=False,
            )
        print()  # Add space so that progress bar is not lost.
        # Update search params.
        start = [search.point(-1, chain=i) for i in search.chains]
        init = None  # Use the termini of the last search without jittering.
        tune = args.nsteps

    if args.logp_out:
        info(f"Writing stochastic search trace to `{args.logp_out}`.")
        trace_logp = np.stack(
            [
                search.model_logp[i * len(search) : (i + 1) * len(search)]
                for i in range(search.nchains)
            ]
        )
        pd.DataFrame(trace_logp).to_csv(
            args.logp_out,
            sep="\t",
            float_format="%0.6f",
            header=False,
            index=False,
        )

    info(f"Optimizing model parameters.")
    mapest1 = find_MAP_loop_retry(model, search)

    info("Optimizing haplotype fingerprints.")
    model.gamma_hyper.set_value(args.fingerprint_ambiguity_penalty)
    start2 = mapest1.copy()
    start2["gamma__interval__"] = model.test_point["gamma__interval__"]
    mapest2 = find_MAP_loop(
        model, start2, vars=[model.gamma, model.epsilon], verbose=2
    )

    info("De-replicating fingerprints.")
    # TODO: Check the logic for this clustering input.
    genotype_vectors = np.cbrt((2 * (mapest2["gamma"][:, :, 0].T)) - 1)
    dmatrix = squareform(pdist(genotype_vectors, metric="minkowski", p=3))
    _clust = (
        AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=args.derep_thresh,
        )
        .fit(dmatrix)
        .labels_
    )

    _clust = pd.Series(_clust)
    nclusts = _clust.unique().shape[0]
    info(f"Initial {args.nstrains} haplotypes reduced to {nclusts} clusters.")
    pi_merged = (
        pd.DataFrame(mapest2["pi"])
        .groupby(_clust, axis="columns")
        .sum()
        .reindex(columns=range(args.nstrains))
        .fillna(1e-6)
        .apply(lambda x: x / x.sum(), axis=1)
    )

    info("Re-fitting fingerprints after haplotype dereplication.")
    start3 = mapest2.copy()
    start3["pi_stickbreaking__"] = stick_breaking.forward(
        pi_merged.values
    ).eval()
    start3["gamma__interval__"] = model.test_point["gamma__interval__"]
    mapest3 = find_MAP_loop(
        model, start3, vars=[model.gamma, model.epsilon], verbose=2
    )

    mapest = mapest3

    # TODO: More principled ambiguity_index calculation.
    ambiguity_score = ambiguity_index(mapest["gamma"], mapest["pi"])
    info(f"Ambiguity index for this fit found to be {ambiguity_score}.")

    frac = pd.DataFrame(mapest["pi"], index=list_samples(pileup_sample))
    frac = frac.loc[:, frac.sum().sort_values(ascending=False).index]
    frac = frac.loc[:, frac.max() >= args.min_max_frac]
    nstrains = frac.shape[1]
    frac.columns = list_strains(nstrains)
    frac["other"] = 1 - frac.sum(1)

    if args.frac_out:
        info(
            f"Writing estimated fractions for {nstrains} strains with "
            f"sufficient max abundance (>{args.min_max_frac}) to "
            f"`{args.frac_out}`."
        )
        (
            frac.stack()[lambda x: x >= args.trunc_frac].to_csv(
                args.frac_out, sep="\t", float_format="%0.6f", header=False
            )
        )

    if args.err_out:
        info(f"Writing estimated sequencing error rates to `{args.err_out}`.")
        err_rate = pd.Series(
            mapest["epsilon"],
            index=list_samples(pileup_sample),
            name="epsilon",
        )
        err_rate.to_csv(
            args.err_out, sep="\t", float_format="%0.6f", header=False
        )

    info(f"Finished")
