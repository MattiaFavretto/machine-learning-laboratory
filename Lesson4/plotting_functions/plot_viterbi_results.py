#!/usr/bin/env python3
"""
plot_viterbi_results.py
Stampa i risultati del punto e) di Exercise 2.4, esportati da viterbi_hmm.py.

  * istogramma degli errori relativi (Viterbi) sulle frasi di test + media
  * stesso grafico per le sequenze casuali di generate_chains (baseline) a confronto

Usage:
    python plot_viterbi_results.py viterbi_results.npz [--output-dir results_plots] [--bins 30] [--show]
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Istogramma errori relativi Viterbi vs sequenze casuali (punto e)")
    p.add_argument("file", nargs="?", default="viterbi_results.npz", help="file .npz esportato da viterbi_hmm.py")
    p.add_argument("--output-dir", "-o", default="results_plots")
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true", help="mostra le finestre (se c'e' un display)")
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.file)
    vit = d["viterbi_errors"]
    rnd = d["random_errors"] if "random_errors" in d.files else np.array([])
    has_rnd = len(rnd) > 0

    print("=" * 70)
    print(f"File: {args.file}  |  h = {int(d['h'])}  |  frasi di test: {len(vit)}")
    print(f"ERRORE RELATIVO MEDIO Viterbi         : {vit.mean():.4f} = {100 * vit.mean():.1f}%")
    if has_rnd:
        print(f"ERRORE RELATIVO MEDIO sequenze casuali: {rnd.mean():.4f} = {100 * rnd.mean():.1f}%"
              f"  ({len(rnd)} frasi)")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Bin comuni, cosi' i due istogrammi sono confrontabili
    allv = np.concatenate([vit, rnd]) if has_rnd else vit
    bins = np.linspace(0, max(allv.max(), 1e-9), args.bins + 1)

    # 1) Viterbi da solo (grafico richiesto dal punto e)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vit, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvline(vit.mean(), color="k", linestyle="--", label=f"mean = {vit.mean():.3f}")
    ax.set_xlabel("Relative error (softmax-weighted Levenshtein / true length)")
    ax.set_ylabel("Number of test utterances")
    ax.set_title("Point e) Relative error of Viterbi decoding on test data")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    f1 = os.path.join(args.output_dir, "viterbi_relative_error_hist")
    fig.savefig(f1 + ".png", dpi=args.dpi)
    fig.savefig(f1 + ".pdf")
    print(f"Salvato: {os.path.abspath(f1)}.png / .pdf")

    # 2) Confronto con la baseline casuale
    if has_rnd:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(vit, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.75, density=True, label="Viterbi")
        ax2.hist(rnd, bins=bins, color="#d62728", edgecolor="black", alpha=0.55, density=True,
                 label="Random sequences (generate_chains)")
        ax2.axvline(vit.mean(), color="#1f77b4", linestyle="--", label=f"Viterbi mean = {vit.mean():.3f}")
        ax2.axvline(rnd.mean(), color="#d62728", linestyle="--", label=f"Random mean = {rnd.mean():.3f}")
        ax2.set_xlabel("Relative error")
        ax2.set_ylabel("Density")
        ax2.set_title("Point e) Viterbi vs random sequences from the estimated model")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        fig2.tight_layout()
        f2 = os.path.join(args.output_dir, "viterbi_vs_random_hist")
        fig2.savefig(f2 + ".png", dpi=args.dpi)
        fig2.savefig(f2 + ".pdf")
        print(f"Salvato: {os.path.abspath(f2)}.png / .pdf")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
