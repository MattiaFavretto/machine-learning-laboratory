#!/usr/bin/env python3
"""
plot_results.py
Analysis and visualization tool for HMM training results (Point c / Exercise 2.4).

Reads trained model parameters (LL history, mean, covariance, transition matrix A)
from an .npz file (or a slurm log file) passed via command line, produces plots of
Log-Likelihood evolution over iterations, and prints formatted matrices.

Usage:
    python plot_results.py [path/to/trained_params.npz]
    python plot_results.py path/to/trained_params.npz --full
    python plot_results.py path/to/project1.log
"""

import argparse
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

# Try to import phoneme names from data_loader_speech if available
PHONEME_NAMES = None
try:
    import data_loader_speech as dl
    _, PHONEME_NAMES = dl.get_phonemes()
except Exception:
    pass

if PHONEME_NAMES is None:
    # Fallback to standard indices [0..41]
    PHONEME_NAMES = [f"Phoneme_{i:02d}" for i in range(40)] + ["START", "END"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisi e visualizzazione risultati EM HMM (Punto c / Esercizio 2.4)"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="trained_params.npz",
        help="Percorso al file con i risultati (.npz salvato da project_1.py, oppure file di log .log/.txt). Default: trained_params.npz",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results_plots",
        help="Directory dove salvare i grafici e i file di testo esportati (default: results_plots)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Mostra interattivamente i grafici a schermo (se l'ambiente grafico è disponibile)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Non aprire finestre grafiche a schermo (modalità headless / cluster)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Stampa per esteso sul terminale tutte le matrici senza troncamento numpy",
    )
    parser.add_argument(
        "--state",
        type=int,
        default=None,
        help="Indice del fonema/stato (0..41) di cui stampare l'intera matrice di covarianza 40x40 sul terminale",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Risoluzione DPI per il salvataggio dei grafici PNG (default: 300)",
    )
    return parser.parse_args()


def match_key(keys, candidates):
    """Case-insensitive search for key matching any candidate."""
    for cand in candidates:
        for k in keys:
            if k.lower() == cand.lower():
                return k
    return None


def parse_log_file(file_path):
    """Extract LL values and optionally matrices from a plain text log file."""
    print(f"Parsing del file di log testuale: {file_path}")
    ll_list = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Match patterns like:
    # "Epoch 1/10 completed - LL: -12345.67"
    # "Epoch 1/10 completed in ... LL: -12345.67"
    regex = re.compile(r"Epoch\s+(\d+)(?:/\d+)?.*?LL:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    for match in regex.finditer(content):
        epoch_idx = int(match.group(1))
        ll_val = float(match.group(2))
        ll_list.append((epoch_idx, ll_val))

    if not ll_list:
        raise ValueError(f"Nessun valore di LL trovato nel file di log {file_path}")

    # Sort by epoch index and take values
    ll_list.sort(key=lambda x: x[0])
    ll = np.array([val for _, val in ll_list])

    return {"ll": ll, "mean": None, "covariance": None, "A": None}


def load_data(file_path):
    """Load results from .npz or .log file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: '{file_path}'")

    if file_path.endswith(".npz"):
        data = np.load(file_path)
        keys = list(data.files)
        print(f"File '{file_path}' caricato con successo. Chiavi presenti: {keys}")

        k_ll = match_key(keys, ["ll", "total_ll_array", "total_ll", "log_likelihood", "lls"])
        k_mean = match_key(keys, ["mean", "means", "mu", "m"])
        k_cov = match_key(keys, ["covariance", "covariances", "cov", "sigma", "covs"])
        k_A = match_key(keys, ["A", "a", "trans_A", "trans", "transition_matrix"])

        ll = np.array(data[k_ll]) if k_ll else None
        mean = np.array(data[k_mean]) if k_mean else None
        covariance = np.array(data[k_cov]) if k_cov else None
        A = np.array(data[k_A]) if k_A else None

        return {"ll": ll, "mean": mean, "covariance": covariance, "A": A}

    elif file_path.endswith((".log", ".txt", ".out")):
        return parse_log_file(file_path)
    else:
        # Try npz first, then log
        try:
            return load_data(file_path + ".npz")
        except Exception:
            return parse_log_file(file_path)


def print_matrices_report(mean, covariance, A, full=False, specific_state=None, output_dir=None):
    """Print matrices to terminal in a neat and readable way, and optionally save them."""
    border = "=" * 80
    sub_border = "-" * 80

    print("\n" + border)
    print("                      STAMPA MATRICI DEL MODELLO HMM")
    print(border)

    if full:
        np.set_printoptions(threshold=np.inf, linewidth=160, precision=4, suppress=True)
    else:
        np.set_printoptions(threshold=100, edgeitems=3, linewidth=120, precision=4, suppress=True)

    # 1. TRANSITION MATRIX A
    if A is not None:
        print("\n" + sub_border)
        print(f"1. MATRICE DI TRANSIZIONE A")
        print(f"   Shape: {A.shape} | Tipo: {A.dtype}")
        print(f"   Valore Min: {np.nanmin(A):.6e} | Valore Max: {np.nanmax(A):.6e}")
        row_sums = np.sum(A, axis=1)
        print(f"   Verifica somme per riga (somma probabilità = 1.0):")
        print(f"     Min somma: {np.nanmin(row_sums):.6f} | Max somma: {np.nanmax(row_sums):.6f} | Media: {np.nanmean(row_sums):.6f}")
        print(sub_border)
        if full:
            print("Matrice A completa:")
            print(A)
        else:
            print("Anteprima Matrice A (usa --full per visualizzare tutti i valori):")
            print(A)
            # Print diagonal self-transition probabilities
            print("\n   Probabilità di auto-transizione A[i, i] (persistenza fonemi):")
            diag_A = np.diag(A)
            for i in range(len(diag_A)):
                name = PHONEME_NAMES[i] if i < len(PHONEME_NAMES) else f"State_{i}"
                print(f"     Stato {i:02d} ({name:10s}): A[{i:02d},{i:02d}] = {diag_A[i]:.4f}")
    else:
        print("\n[INFO] Matrice A non presente nel file caricato.")

    # 2. MEAN MATRIX
    if mean is not None:
        print("\n" + sub_border)
        print(f"2. MATRICE DELLE MEDIE (Emission Means)")
        print(f"   Shape: {mean.shape} | (Numero Fonemi x Feature MFCC)")
        print(f"   Valore Min: {np.nanmin(mean):.6f} | Valore Max: {np.nanmax(mean):.6f}")
        print(f"   Media Globale: {np.nanmean(mean):.6f} | Deviazione Std Globale: {np.nanstd(mean):.6f}")
        print(sub_border)
        if full:
            print("Matrice delle Medie completa:")
            print(mean)
        else:
            print("Anteprima Matrice delle Medie:")
            print(mean)
            print("\n   Medie dei primi 5 e ultimi 5 fonemi (prime 6 feature MFCC ciascuno):")
            indices = list(range(min(5, len(mean))))
            if len(mean) > 10:
                indices += list(range(len(mean) - 5, len(mean)))
            for i in indices:
                name = PHONEME_NAMES[i] if i < len(PHONEME_NAMES) else f"State_{i}"
                feat_preview = ", ".join(f"{v:7.2f}" for v in mean[i, :6])
                print(f"     Stato {i:02d} ({name:10s}): [{feat_preview}, ...]")
    else:
        print("\n[INFO] Matrice delle medie non presente nel file caricato.")

    # 3. COVARIANCE MATRICES
    if covariance is not None:
        print("\n" + sub_border)
        print(f"3. MATRICI DI COVARIANZA (Emission Covariances)")
        print(f"   Shape: {covariance.shape} | (Numero Fonemi x Feature x Feature)")
        print(f"   Valore Min: {np.nanmin(covariance):.6e} | Valore Max: {np.nanmax(covariance):.6e}")
        
        # Variances (diagonals)
        variances = np.array([np.diag(covariance[k]) for k in range(len(covariance))])
        print(f"   Varianze diagonali (min: {np.nanmin(variances):.6e}, max: {np.nanmax(variances):.6e}, media: {np.nanmean(variances):.6e})")
        print(sub_border)

        if specific_state is not None:
            if 0 <= specific_state < len(covariance):
                name = PHONEME_NAMES[specific_state] if specific_state < len(PHONEME_NAMES) else f"State_{specific_state}"
                print(f"\nMatrice di covarianza completa per lo stato {specific_state} ({name}):")
                print(covariance[specific_state])
            else:
                print(f"[ATTENZIONE] Indice stato {specific_state} non valido (deve essere tra 0 e {len(covariance)-1}).")
        elif full:
            print("Tutte le matrici di covarianza (per ciascun fonema):")
            for k in range(len(covariance)):
                name = PHONEME_NAMES[k] if k < len(PHONEME_NAMES) else f"State_{k}"
                print(f"\n--- Covarianza Stato {k:02d} ({name}) ---")
                print(covariance[k])
        else:
            print("Anteprima Matrice di Covarianza dello Stato 0:")
            name0 = PHONEME_NAMES[0] if len(PHONEME_NAMES) > 0 else "State_00"
            print(f"(Stato 0: {name0})")
            print(covariance[0])
            print("\n   [SUGGERIMENTO] Per vedere la matrice di covarianza 40x40 di un fonema specifico:")
            print("       python plot_results.py --state <indice_fonema>")
            print("   Per stampare tutte le 42 matrici di covarianza sul terminale:")
            print("       python plot_results.py --full")
    else:
        print("\n[INFO] Matrice di covarianza non presente nel file caricato.")

    # Save to disk as clean text files
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if A is not None:
            np.savetxt(os.path.join(output_dir, "matrix_A.csv"), A, delimiter=",", fmt="%.8f")
            with open(os.path.join(output_dir, "matrix_A.txt"), "w") as f:
                f.write(f"# Matrice di Transizione A ({A.shape[0]}x{A.shape[1]})\n")
                np.savetxt(f, A, fmt="%12.6f")
        if mean is not None:
            np.savetxt(os.path.join(output_dir, "matrix_mean.csv"), mean, delimiter=",", fmt="%.8f")
            with open(os.path.join(output_dir, "matrix_mean.txt"), "w") as f:
                f.write(f"# Matrice delle Medie ({mean.shape[0]}x{mean.shape[1]})\n")
                np.savetxt(f, mean, fmt="%12.6f")
        if covariance is not None:
            with open(os.path.join(output_dir, "matrix_covariances.txt"), "w") as f:
                f.write(f"# Matrice delle Covarianze ({covariance.shape[0]}x{covariance.shape[1]}x{covariance.shape[2]})\n")
                for k in range(len(covariance)):
                    name = PHONEME_NAMES[k] if k < len(PHONEME_NAMES) else f"State_{k}"
                    f.write(f"\n# --- STATO {k:02d}: {name} ---\n")
                    np.savetxt(f, covariance[k], fmt="%12.6f")
            # Save diagonal variances summary
            np.savetxt(os.path.join(output_dir, "covariances_diagonal_variances.csv"), variances, delimiter=",", fmt="%.8f")

        print(f"\n[OK] Copie testuali complete salvate in: {os.path.abspath(output_dir)}")
    print(border + "\n")


def plot_ll_evolution(ll, output_dir, show=False, dpi=300):
    """Plot evolution of Log-Likelihood over iterations (Point c / Exercise 2.4)."""
    if ll is None or len(ll) == 0:
        print("[AVVISO] Nessun array di Log-Likelihood trovato per generare il grafico.")
        return

    epochs = np.arange(1, len(ll) + 1)

    print("\n" + "=" * 80)
    print("                    PROGRESSIONE LOG-LIKELIHOOD (LL)")
    print("=" * 80)
    print(f"{'Epoca':<10} {'Log-Likelihood':<22} {'Delta LL (Miglioramento)':<25} {'Delta Relativo'}")
    print("-" * 80)
    for i, val in enumerate(ll):
        if i == 0:
            print(f"{i + 1:<10} {val:<22.4f} {'---':<25} {'---'}")
        else:
            delta = val - ll[i - 1]
            rel = (delta / abs(ll[i - 1])) * 100 if ll[i - 1] != 0 else 0
            delta_str = f"{delta:+.4f}"
            rel_str = f"{rel:+.4f}%"
            print(f"{i + 1:<10} {val:<22.4f} {delta_str:<25} {rel_str}")
    print("=" * 80)

    # Setup matplotlib style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
    })

    # Create figure with 2 subplots:
    # 1) Absolute Log-Likelihood over epochs
    # 2) Step improvement (delta LL) showing EM convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: LL evolution
    ax1.plot(epochs, ll, marker="o", color="#1f77b4", linewidth=2.2, markersize=7, label="Log-Likelihood (LL)")
    ax1.set_xlabel("Iterazione EM (Epoca)")
    ax1.set_ylabel("Log-Likelihood Totale")
    ax1.set_title("Evoluzione della Log-Likelihood nel tempo (Punto c)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(epochs)

    # Annotate start and end points
    ax1.annotate(
        f"Inizio: {ll[0]:.2e}",
        xy=(epochs[0], ll[0]),
        xytext=(epochs[0] + 0.2, ll[0]),
        fontsize=9,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    )
    ax1.annotate(
        f"Finale: {ll[-1]:.2e}",
        xy=(epochs[-1], ll[-1]),
        xytext=(epochs[-1] - 0.4, ll[-1]),
        ha="right",
        fontsize=9,
        fontweight="bold",
        color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
    )
    ax1.legend(loc="lower right")

    # Subplot 2: Improvement Delta LL = LL_t - LL_{t-1}
    if len(ll) > 1:
        deltas = np.diff(ll)
        delta_epochs = epochs[1:]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax2.bar(delta_epochs, deltas, color=colors, width=0.5, alpha=0.85, edgecolor="black")
        ax2.set_xlabel("Iterazione EM (Epoca)")
        ax2.set_ylabel(r"$\Delta$ Log-Likelihood ($LL_t - LL_{t-1}$)")
        ax2.set_title("Miglioramento per iterazione (Convergenza EM)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.set_xticks(delta_epochs)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)

        # Annotate total improvement
        total_delta = ll[-1] - ll[0]
        ax2.text(
            0.05,
            0.92,
            f"Miglioramento totale:\n+{total_delta:.4e}" if total_delta >= 0 else f"Variazione totale:\n{total_delta:.4e}",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            fontsize=10,
        )
    else:
        ax2.text(0.5, 0.5, "Almeno 2 epoche necessarie per calcolare Delta LL", ha="center", va="center")

    plt.tight_layout()

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "ll_evolution.png")
    pdf_path = os.path.join(output_dir, "ll_evolution.pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\n[OK] Grafici salvati:")
    print(f"     PNG (alta risoluzione): {os.path.abspath(png_path)}")
    print(f"     PDF (vettoriale per LaTeX): {os.path.abspath(pdf_path)}")

    # Standalone single plot for clean report insertion
    fig_single, ax_s = plt.subplots(figsize=(8, 5))
    ax_s.plot(epochs, ll, marker="s", color="#0055aa", linewidth=2.0, markersize=6, label="Log-Likelihood")
    ax_s.set_xlabel("Iteration (Epoch)")
    ax_s.set_ylabel("Total Log-Likelihood")
    ax_s.set_title("Evolution of Log-Likelihood over EM Iterations")
    ax_s.grid(True, linestyle="--", alpha=0.7)
    ax_s.set_xticks(epochs)
    ax_s.legend()
    plt.tight_layout()
    single_png = os.path.join(output_dir, "ll_evolution_single.png")
    single_pdf = os.path.join(output_dir, "ll_evolution_single.pdf")
    plt.savefig(single_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(single_pdf, bbox_inches="tight")
    plt.close(fig_single)

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[INFO] Impossibile aprire display grafico interattivo ({e}). Grafici salvati su file.")
    else:
        plt.close(fig)


def plot_heatmaps(A, mean, covariance, output_dir, dpi=300):
    """Generate visual heatmaps of Matrix A, Mean vectors, and Covariance variances."""
    if A is None and mean is None and covariance is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Transition Matrix A Heatmap
    if A is not None:
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(A, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Probabilità di Transizione $A_{ij}$")
        ax.set_title("Matrice di Transizione A ($K \\times K$)")
        ax.set_xlabel("Stato Successivo $j$")
        ax.set_ylabel("Stato Corrente $i$")

        if len(A) <= 42:
            step = 2 if len(A) > 20 else 1
            ticks = list(range(0, len(A), step))
            labels = [PHONEME_NAMES[t] if t < len(PHONEME_NAMES) else str(t) for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)

        plt.tight_layout()
        a_png = os.path.join(output_dir, "matrix_A_heatmap.png")
        a_pdf = os.path.join(output_dir, "matrix_A_heatmap.pdf")
        plt.savefig(a_png, dpi=dpi, bbox_inches="tight")
        plt.savefig(a_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"     Heatmap Matrice A: {os.path.abspath(a_png)}")

    # 2. Mean Matrix Heatmap
    if mean is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mean, cmap="coolwarm", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Valore Medio $\\mu_{kd}$")
        ax.set_title("Matrice delle Medie per Fonema $\\mu$ ($K \\times D$)")
        ax.set_xlabel("Indice Feature MFCC ($d = 0 \\dots 39$)")
        ax.set_ylabel("Fonema / Stato $k$")

        if len(mean) <= 42:
            step = 2 if len(mean) > 20 else 1
            ticks = list(range(0, len(mean), step))
            labels = [PHONEME_NAMES[t] if t < len(PHONEME_NAMES) else str(t) for t in ticks]
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)

        plt.tight_layout()
        m_png = os.path.join(output_dir, "matrix_mean_heatmap.png")
        m_pdf = os.path.join(output_dir, "matrix_mean_heatmap.pdf")
        plt.savefig(m_png, dpi=dpi, bbox_inches="tight")
        plt.savefig(m_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"     Heatmap Medie: {os.path.abspath(m_png)}")

    # 3. Covariance Diagonals (Variances) Heatmap
    if covariance is not None:
        variances = np.array([np.diag(covariance[k]) for k in range(len(covariance))])
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(np.log10(np.maximum(variances, 1e-12)), cmap="inferno", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\log_{10}(\sigma^2_{kd})$ (Varianza)")
        ax.set_title(r"Varianze Diagonali delle Emissioni ($\log_{10} \mathrm{diag}(\Sigma_k)$)")
        ax.set_xlabel("Indice Feature MFCC ($d = 0 \\dots 39$)")
        ax.set_ylabel("Fonema / Stato $k$")

        if len(variances) <= 42:
            step = 2 if len(variances) > 20 else 1
            ticks = list(range(0, len(variances), step))
            labels = [PHONEME_NAMES[t] if t < len(PHONEME_NAMES) else str(t) for t in ticks]
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)

        plt.tight_layout()
        v_png = os.path.join(output_dir, "matrix_covariance_variances_heatmap.png")
        v_pdf = os.path.join(output_dir, "matrix_covariance_variances_heatmap.pdf")
        plt.savefig(v_png, dpi=dpi, bbox_inches="tight")
        plt.savefig(v_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"     Heatmap Varianze: {os.path.abspath(v_png)}")


def main():
    args = parse_args()
    file_path = args.file

    # If file doesn't exist directly, check inside current working directory or Lesson4
    if not os.path.exists(file_path):
        alt_paths = [
            os.path.join("Lesson4", file_path),
            os.path.join("..", file_path),
            os.path.join(os.path.dirname(__file__), file_path),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                file_path = alt
                break

    try:
        data = load_data(file_path)
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare il file '{file_path}': {e}", file=sys.stderr)
        print("\nUso:", file=sys.stderr)
        print("    python plot_results.py <percorso/al/file_risultati.npz>", file=sys.stderr)
        print("    python plot_results.py <percorso/al/file_log.log>", file=sys.stderr)
        sys.exit(1)

    ll = data.get("ll")
    mean = data.get("mean")
    covariance = data.get("covariance")
    A = data.get("A")

    # Output directory
    output_dir = args.output_dir

    # 1. Print Matrices Report
    print_matrices_report(
        mean=mean,
        covariance=covariance,
        A=A,
        full=args.full,
        specific_state=args.state,
        output_dir=output_dir,
    )

    # 2. Plot Log-Likelihood Evolution (Point c / Point 3)
    has_display = bool(os.environ.get("DISPLAY"))
    should_show = args.show or (has_display and not args.no_show and not args.show)

    plot_ll_evolution(ll=ll, output_dir=output_dir, show=args.show, dpi=args.dpi)

    # 3. Generate Visual Heatmaps of the Matrices
    print("\n[INFO] Generazione heatmap visive per le matrici...")
    plot_heatmaps(A=A, mean=mean, covariance=covariance, output_dir=output_dir, dpi=args.dpi)

    print("\nElaborazione completata con successo!\n")


if __name__ == "__main__":
    main()
