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

# Standard 40 CMU Sphinx phonemes + START + END with ARPAbet and IPA transcription
DEFAULT_SPHINX_PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL",
    "START", "END"
]

DEFAULT_IPA_MAPPING = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ",
    "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
    "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ", "SIL": "sil",
    "START": "start", "END": "end"
}

PHONEME_NAMES = DEFAULT_SPHINX_PHONEMES
PHONEME_IPA = [DEFAULT_IPA_MAPPING.get(p, p) for p in DEFAULT_SPHINX_PHONEMES]

try:
    import data_loader_speech as dl
    _, loaded_phones = dl.get_phonemes()
    if loaded_phones is not None and len(loaded_phones) > 0:
        PHONEME_NAMES = loaded_phones
        try:
            ipa_dict = {a[0]: a[1] for a in dl.read_csv('/share/datasets/LJSpeech-1.1/sphinxdict/toIPA.csv')}
            ipa_dict["SIL"] = "sil"
            ipa_dict["START"] = "start"
            ipa_dict["END"] = "end"
            PHONEME_IPA = [ipa_dict.get(p, DEFAULT_IPA_MAPPING.get(p, p)) for p in PHONEME_NAMES]
        except Exception:
            PHONEME_IPA = [DEFAULT_IPA_MAPPING.get(p, p) for p in PHONEME_NAMES]
except Exception:
    pass


def get_phoneme_display(idx, phonemes=None, ipa=None, style="full"):
    """
    Returns a clean, human-readable string for phoneme state idx.
    style:
      - 'full':  e.g. "'AA' (/ɑ/)"
      - 'short': e.g. "AA /ɑ/"
      - 'name':  e.g. "AA"
      - 'ipa':   e.g. "/ɑ/"
    """
    name = None
    if phonemes is not None and idx < len(phonemes):
        name = str(phonemes[idx])
    elif idx < len(PHONEME_NAMES):
        name = PHONEME_NAMES[idx]
    else:
        name = f"State_{idx:02d}"

    val_ipa = None
    if ipa is not None and idx < len(ipa):
        val_ipa = str(ipa[idx])
    elif idx < len(PHONEME_IPA):
        val_ipa = PHONEME_IPA[idx]
    elif name in DEFAULT_IPA_MAPPING:
        val_ipa = DEFAULT_IPA_MAPPING[name]

    if val_ipa and str(val_ipa).strip() and str(val_ipa).strip() != name:
        ipa_str = f"/{val_ipa}/"
        if style == "ipa":
            return ipa_str
        elif style == "name":
            return name
        elif style == "short":
            return f"{name} {ipa_str}"
        else:  # full
            return f"'{name}' ({ipa_str})"
    else:
        return f"'{name}'" if style == "full" else name


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
        "--top-k",
        type=int,
        default=5,
        help="Numero di coppie di fonemi più/meno probabili da confondere da mostrare (Punto d, default: 5)",
    )
    parser.add_argument(
        "--include-boundary",
        action="store_true",
        default=False,
        help="Includi gli stati START ed END nell'analisi di confusione fonetica (default: False, analizza i 40 fonemi)",
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
        k_ph = match_key(keys, ["phonemes", "phoneme_names", "phonemes_list", "phones", "names"])
        k_ipa = match_key(keys, ["phonemes_ipa", "ipa", "transcriptions", "ipa_transcriptions", "ipa_dict"])

        ll = np.array(data[k_ll]) if k_ll else None
        mean = np.array(data[k_mean]) if k_mean else None
        covariance = np.array(data[k_cov]) if k_cov else None
        A = np.array(data[k_A]) if k_A else None
        phonemes = [str(x) for x in data[k_ph]] if k_ph else None
        phonemes_ipa = [str(x) for x in data[k_ipa]] if k_ipa else None

        return {
            "ll": ll,
            "mean": mean,
            "covariance": covariance,
            "A": A,
            "phonemes": phonemes,
            "phonemes_ipa": phonemes_ipa,
        }

    elif file_path.endswith((".log", ".txt", ".out")):
        parsed = parse_log_file(file_path)
        parsed["phonemes"] = None
        parsed["phonemes_ipa"] = None
        return parsed
    else:
        # Try npz first, then log
        try:
            return load_data(file_path + ".npz")
        except Exception:
            parsed = parse_log_file(file_path)
            parsed["phonemes"] = None
            parsed["phonemes_ipa"] = None
            return parsed


def print_matrices_report(mean, covariance, A, phonemes=None, phonemes_ipa=None, full=False, specific_state=None, output_dir=None):
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
                label = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
                print(f"     Stato {i:02d} ({label:<20s}): A[{i:02d},{i:02d}] = {diag_A[i]:.4f}")
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
                label = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
                feat_preview = ", ".join(f"{v:7.2f}" for v in mean[i, :6])
                print(f"     Stato {i:02d} ({label:<20s}): [{feat_preview}, ...]")
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
                label = get_phoneme_display(specific_state, phonemes, phonemes_ipa, style="full")
                print(f"\nMatrice di covarianza completa per lo stato {specific_state} ({label}):")
                print(covariance[specific_state])
            else:
                print(f"[ATTENZIONE] Indice stato {specific_state} non valido (deve essere tra 0 e {len(covariance)-1}).")
        elif full:
            print("Tutte le matrici di covarianza (per ciascun fonema):")
            for k in range(len(covariance)):
                label = get_phoneme_display(k, phonemes, phonemes_ipa, style="full")
                print(f"\n--- Covarianza Stato {k:02d} ({label}) ---")
                print(covariance[k])
        else:
            print("Anteprima Matrice di Covarianza dello Stato 0:")
            label0 = get_phoneme_display(0, phonemes, phonemes_ipa, style="full")
            print(f"(Stato 0: {label0})")
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
                    label = get_phoneme_display(k, phonemes, phonemes_ipa, style="full")
                    f.write(f"\n# --- STATO {k:02d}: {label} ---\n")
                    np.savetxt(f, covariance[k], fmt="%12.6f")
            # Save diagonal variances summary
            np.savetxt(os.path.join(output_dir, "covariances_diagonal_variances.csv"), variances, delimiter=",", fmt="%.8f")

        print(f"\n[OK] Copie testuali complete salvate in: {os.path.abspath(output_dir)}")
    print(border + "\n")


def compute_symmetrized_kl_matrix(mean, covariance, eps=1e-6):
    """
    Computes the full K x K pairwise Symmetrized Kullback-Leibler (KL) Divergence
    between multivariate Gaussian emission distributions N(mu_i, Sigma_i).

    Formula for multivariate Gaussians:
        D_sym(i, j) = D_KL(i || j) + D_KL(j || i)
                    = 0.5 * [ tr(Sigma_j^-1 Sigma_i) + tr(Sigma_i^-1 Sigma_j)
                             + (mu_i - mu_j)^T (Sigma_i^-1 + Sigma_j^-1) (mu_i - mu_j) - 2*D ]
    """
    K, D = mean.shape
    eye = np.eye(D)
    inv_covs = np.empty_like(covariance)
    for k in range(K):
        cov_reg = covariance[k] + eps * eye
        inv_covs[k] = np.linalg.inv(cov_reg)

    kl_matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            diff = mean[i] - mean[j]
            trace_term = np.trace(inv_covs[j] @ covariance[i]) + np.trace(inv_covs[i] @ covariance[j]) - 2.0 * D
            quad_term = float(diff.T @ (inv_covs[i] + inv_covs[j]) @ diff)
            d_sym = 0.5 * (trace_term + quad_term)
            kl_matrix[i, j] = d_sym
            kl_matrix[j, i] = d_sym
    return kl_matrix


def plot_confusion_pairs(
    most_confusable,
    least_confusable,
    phonemes=None,
    phonemes_ipa=None,
    output_dir="results_plots",
    show=False,
    dpi=300,
):
    """
    Plots a dual-panel horizontal bar chart comparing the top most confusable pairs
    (minimum symmetrised KL divergence) and least confusable pairs (maximum KL divergence).
    """
    if not most_confusable and not least_confusable:
        return

    top_k = max(len(most_confusable), len(least_confusable))
    fig_height = max(5.5, 0.75 * top_k + 2.0)
    fig, (ax_most, ax_least) = plt.subplots(1, 2, figsize=(16.5, fig_height))

    def format_val(v):
        if abs(v) >= 1e4 or (0 < abs(v) < 1e-2):
            return f"{v:.2e}"
        elif abs(v) >= 100:
            return f"{v:.1f}"
        else:
            return f"{v:.4f}"

    # Panel 1: Most Confusable (Min KL Divergence)
    if most_confusable:
        y_pos = np.arange(len(most_confusable))
        most_vals = [p[0] for p in most_confusable]
        most_labels = [
            f"#{r}  [S{p[1]:02d}-S{p[2]:02d}]  {get_phoneme_display(p[1], phonemes, phonemes_ipa, style='short')}  ↔  {get_phoneme_display(p[2], phonemes, phonemes_ipa, style='short')}"
            for r, p in enumerate(most_confusable, 1)
        ]
        colors_most = plt.cm.OrRd(np.linspace(0.85, 0.45, len(most_confusable)))
        bars_m = ax_most.barh(y_pos, most_vals, color=colors_most, edgecolor="#333333", height=0.6, alpha=0.9)
        ax_most.set_yticks(y_pos)
        ax_most.set_yticklabels(most_labels, fontsize=10)
        ax_most.invert_yaxis()
        ax_most.set_xlabel(r"Symmetrized KL Divergence $D_{\mathrm{sym}}$ (linear scale)", fontsize=11)
        ax_most.set_title(
            f"Top {len(most_confusable)} Most Confusable Phoneme Pairs\n(Min KL Divergence $\\rightarrow$ High Acoustic Overlap)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax_most.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_most.spines["top"].set_visible(False)
        ax_most.spines["right"].set_visible(False)
        max_m = max(most_vals) if most_vals else 1.0
        ax_most.set_xlim(0, max_m * 1.35 if max_m > 0 else 1.0)

        for bar, val in zip(bars_m, most_vals):
            w = bar.get_width()
            ax_most.text(
                w + max_m * 0.02,
                bar.get_y() + bar.get_height() / 2,
                format_val(val),
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#800000",
            )
    else:
        ax_most.text(0.5, 0.5, "Nessun dato sulle coppie più confondibili", ha="center", va="center")

    # Panel 2: Least Confusable (Max KL Divergence)
    if least_confusable:
        y_pos_l = np.arange(len(least_confusable))
        least_vals = [p[0] for p in least_confusable]
        least_labels = [
            f"#{r}  [S{p[1]:02d}-S{p[2]:02d}]  {get_phoneme_display(p[1], phonemes, phonemes_ipa, style='short')}  ↔  {get_phoneme_display(p[2], phonemes, phonemes_ipa, style='short')}"
            for r, p in enumerate(least_confusable, 1)
        ]
        colors_least = plt.cm.Blues(np.linspace(0.85, 0.45, len(least_confusable)))
        bars_l = ax_least.barh(y_pos_l, least_vals, color=colors_least, edgecolor="#333333", height=0.6, alpha=0.9)
        ax_least.set_yticks(y_pos_l)
        ax_least.set_yticklabels(least_labels, fontsize=10)
        ax_least.invert_yaxis()

        min_l = min(least_vals)
        max_l = max(least_vals)
        use_log = (max_l / max(min_l, 1e-6) >= 10.0) or (max_l > 100.0)
        if use_log:
            ax_least.set_xscale("log")
            ax_least.set_xlim(min_l * 0.5 if min_l > 0 else 1e-1, max_l * 8.0)
            scale_str = r"$\log_{10}$ scale"
        else:
            ax_least.set_xlim(0, max_l * 1.35 if max_l > 0 else 1.0)
            scale_str = "linear scale"

        ax_least.set_xlabel(f"Symmetrized KL Divergence $D_{{\\mathrm{{sym}}}}$ ({scale_str})", fontsize=11)
        ax_least.set_title(
            f"Top {len(least_confusable)} Least Confusable Phoneme Pairs\n(Max KL Divergence $\\rightarrow$ Distinct Distributions)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax_least.grid(True, axis="x", linestyle="--", alpha=0.5, which="both" if use_log else "major")
        ax_least.spines["top"].set_visible(False)
        ax_least.spines["right"].set_visible(False)

        for bar, val in zip(bars_l, least_vals):
            w = bar.get_width()
            offset = w * 1.15 if use_log else (w + max_l * 0.02)
            ax_least.text(
                offset,
                bar.get_y() + bar.get_height() / 2,
                format_val(val),
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#003366",
            )
    else:
        ax_least.text(0.5, 0.5, "Nessun dato sulle coppie meno confondibili", ha="center", va="center")

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        p_png = os.path.join(output_dir, "phoneme_confusion_pairs.png")
        p_pdf = os.path.join(output_dir, "phoneme_confusion_pairs.pdf")
        fig.savefig(p_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(p_pdf, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)


def plot_confusion_mfcc_profiles(
    most_confusable,
    least_confusable,
    mean,
    covariance,
    phonemes=None,
    phonemes_ipa=None,
    output_dir="results_plots",
    show=False,
    dpi=300,
):
    """
    Plots MFCC emission profiles (mean +/- 1 std) across the feature dimensions
    for the #1 Most Confusable pair (high overlap) and #1 Least Confusable pair (wide separation).
    Provides physical intuition into why the KL divergence is small or large.
    """
    if mean is None or covariance is None or not most_confusable or not least_confusable:
        return

    D = mean.shape[1]
    feats = np.arange(D)
    variances = np.array([np.diag(covariance[k]) for k in range(len(covariance))])
    std = np.sqrt(np.maximum(variances, 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    # Top: #1 Most Confusable
    d_m, i_m, j_m = most_confusable[0]
    lbl_m_i = get_phoneme_display(i_m, phonemes, phonemes_ipa, style="short")
    lbl_m_j = get_phoneme_display(j_m, phonemes, phonemes_ipa, style="short")

    ax1.plot(feats, mean[i_m], color="#d95f02", lw=2.2, marker="o", markersize=4, label=f"Stato {i_m:02d}: {lbl_m_i} (Media)")
    ax1.fill_between(feats, mean[i_m] - std[i_m], mean[i_m] + std[i_m], color="#d95f02", alpha=0.22, label=f"{lbl_m_i} (±1σ)")
    ax1.plot(feats, mean[j_m], color="#7570b3", lw=2.2, marker="s", markersize=4, linestyle="--", label=f"Stato {j_m:02d}: {lbl_m_j} (Media)")
    ax1.fill_between(feats, mean[j_m] - std[j_m], mean[j_m] + std[j_m], color="#7570b3", alpha=0.22, label=f"{lbl_m_j} (±1σ)")
    ax1.set_title(
        f"Acoustic MFCC Comparison: Top Confusable Pair ({lbl_m_i} vs {lbl_m_j}) — $D_{{\\mathrm{{sym}}}} = {d_m:.4f}$\n"
        r"(Distributions heavily overlap $\rightarrow$ High acoustic confusion risk in speech decoding)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("MFCC Feature Value", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", ncol=2, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: #1 Least Confusable
    d_l, i_l, j_l = least_confusable[0]
    lbl_l_i = get_phoneme_display(i_l, phonemes, phonemes_ipa, style="short")
    lbl_l_j = get_phoneme_display(j_l, phonemes, phonemes_ipa, style="short")

    ax2.plot(feats, mean[i_l], color="#1b9e77", lw=2.2, marker="o", markersize=4, label=f"Stato {i_l:02d}: {lbl_l_i} (Media)")
    ax2.fill_between(feats, mean[i_l] - std[i_l], mean[i_l] + std[i_l], color="#1b9e77", alpha=0.22, label=f"{lbl_l_i} (±1σ)")
    ax2.plot(feats, mean[j_l], color="#e7298a", lw=2.2, marker="^", markersize=4, linestyle="--", label=f"Stato {j_l:02d}: {lbl_l_j} (Media)")
    ax2.fill_between(feats, mean[j_l] - std[j_l], mean[j_l] + std[j_l], color="#e7298a", alpha=0.22, label=f"{lbl_l_j} (±1σ)")
    ax2.set_title(
        f"Acoustic MFCC Comparison: Least Confusable Pair ({lbl_l_i} vs {lbl_l_j}) — $D_{{\\mathrm{{sym}}}} = {d_l:.2e}$\n"
        r"(Distributions are drastically separated $\rightarrow$ Negligible confusion probability)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("MFCC Feature Dimension Index ($d = 0 \\dots 39$)", fontsize=10)
    ax2.set_ylabel("MFCC Feature Value", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", ncol=2, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prof_png = os.path.join(output_dir, "phoneme_confusion_mfcc_profiles.png")
        prof_pdf = os.path.join(output_dir, "phoneme_confusion_mfcc_profiles.pdf")
        fig.savefig(prof_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(prof_pdf, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)


def plot_confusion_submatrix(
    most_confusable,
    kl_matrix,
    phonemes=None,
    phonemes_ipa=None,
    output_dir="results_plots",
    show=False,
    dpi=300,
):
    """
    Plots a focused, annotated heatmap showing pairwise symmetric KL divergences
    among the phonemes involved in the top most confusable pairs.
    """
    if not most_confusable or kl_matrix is None:
        return

    involved = sorted(list(set([p[1] for p in most_confusable] + [p[2] for p in most_confusable])))
    if len(involved) < 2:
        return

    sub_kl = kl_matrix[np.ix_(involved, involved)]
    labels = [get_phoneme_display(idx, phonemes, phonemes_ipa, style="short") for idx in involved]
    n = len(involved)

    fig_dim = max(7.0, 0.8 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim - 0.5))

    im = ax.imshow(sub_kl, cmap="YlOrRd_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Symmetrized KL Divergence $D_{\mathrm{sym}}$ (lower = more confusable)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)

    min_val, max_val = sub_kl.min(), sub_kl.max()
    mid_val = 0.5 * (min_val + max_val)

    for r in range(n):
        for c in range(n):
            val = sub_kl[r, c]
            txt_color = "white" if val > mid_val else "black"
            txt = f"{val:.1f}" if val >= 10 else f"{val:.2f}"
            ax.text(c, r, txt, ha="center", va="center", color=txt_color, fontsize=8.5, fontweight="bold")

    ax.set_title("Focused Confusion Matrix: Acoustic Proximity of Top Confusable Phonemes", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        sub_png = os.path.join(output_dir, "phoneme_confusion_submatrix.png")
        sub_pdf = os.path.join(output_dir, "phoneme_confusion_submatrix.pdf")
        fig.savefig(sub_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(sub_pdf, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)


def compute_and_print_kl_confusion(
    mean,
    covariance,
    phonemes=None,
    phonemes_ipa=None,
    top_k=5,
    include_boundary=False,
    output_dir=None,
    show=False,
    dpi=300,
):
    """
    Sanity check (Point d / Exercise 2.4):
    Calculates the top-k most probable and top-k least probable phoneme confusion pairs
    using the Symmetrized Kullback-Leibler Divergence between emission distributions.

    - Most Probable Confusion Pairs: MINIMUM KL divergence (distributions heavily overlap).
    - Least Probable Confusion Pairs: MAXIMUM KL divergence (distributions far apart).
    """
    if mean is None or covariance is None:
        print("\n[INFO] Matrice delle medie o covarianze assente: calcolo divergenza KL non eseguibile.")
        return None

    if phonemes is None:
        phonemes = PHONEME_NAMES
    if phonemes_ipa is None:
        phonemes_ipa = PHONEME_IPA

    K = len(mean)
    # Determine candidate indices: optionally exclude START (state 40) and END (state 41) if K > 40
    if not include_boundary and K > 40:
        candidate_indices = [i for i in range(min(40, K))]
        scope_str = f"primi {len(candidate_indices)} fonemi effettivi (esclusi START ed END)"
    else:
        candidate_indices = list(range(K))
        scope_str = f"tutti i {K} stati/fonemi"

    kl_matrix = compute_symmetrized_kl_matrix(mean, covariance)

    # Collect unique pairs (i, j) with i < j
    pairs = []
    for idx_a, i in enumerate(candidate_indices):
        for j in candidate_indices[idx_a + 1:]:
            d_sym = kl_matrix[i, j]
            pairs.append((d_sym, i, j))

    pairs.sort(key=lambda x: x[0])
    most_confusable = pairs[:top_k]
    least_confusable = pairs[-top_k:][::-1]

    border = "=" * 80
    sub_border = "-" * 80

    print("\n" + border)
    print("      ANALISI CONFUSIONE FONEMI CON SYMMETRISED KL-DIVERGENCE (PUNTO d)")
    print(border)
    print(f"Ambito di calcolo: {scope_str}")
    print("Definizione metrica: D_sym(i, j) = D_KL(p_i || p_j) + D_KL(p_j || p_i)")
    print("Interpretazione nel riconoscimento del parlato:")
    print("  * MINIMA divergenza KL  ==> Massima probabilità di confusione (distribuzioni acustiche sovrapposte)")
    print("  * MASSIMA divergenza KL ==> Minima probabilità di confusione (distribuzioni acustiche ben separate)")
    print(sub_border)

    # 1. Most Probable Confusion Pairs
    print(f"\n>>> 1. LE {top_k} COPPIE CON CUI E' PIU' PROBABILE CONFONDERSI (Minima Divergenza KL):")
    print(f"{'Rank':<6} {'Coppia Stati':<22} {'Trascrizione Fonemi':<32} {'Divergenza KL Simmetrica'}")
    print(sub_border)
    for rank, (d_val, i, j) in enumerate(most_confusable, 1):
        disp_i = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
        disp_j = get_phoneme_display(j, phonemes, phonemes_ipa, style="full")
        pair_states = f"Stato {i:02d} - Stato {j:02d}"
        pair_names = f"{disp_i} <-> {disp_j}"
        print(f"#{rank:<5} {pair_states:<22} {pair_names:<32} {d_val:<25.4f}")

    # 2. Least Probable Confusion Pairs
    print(f"\n>>> 2. LE {top_k} COPPIE CON CUI E' MENO PROBABILE CONFONDERSI (Massima Divergenza KL):")
    print(f"{'Rank':<6} {'Coppia Stati':<22} {'Trascrizione Fonemi':<32} {'Divergenza KL Simmetrica'}")
    print(sub_border)
    for rank, (d_val, i, j) in enumerate(least_confusable, 1):
        disp_i = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
        disp_j = get_phoneme_display(j, phonemes, phonemes_ipa, style="full")
        pair_states = f"Stato {i:02d} - Stato {j:02d}"
        pair_names = f"{disp_i} <-> {disp_j}"
        print(f"#{rank:<5} {pair_states:<22} {pair_names:<32} {d_val:<25.4e}")

    print("\n" + border)

    # Export to files
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, "phoneme_confusion_pairs.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("      ANALISI CONFUSIONE FONEMI CON SYMMETRISED KL-DIVERGENCE (Punto d)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Ambito di calcolo: {scope_str}\n\n")
            f.write(f"1. TOP {top_k} COPPIE CON CUI E' PIU' PROBABILE CONFONDERSI (Minima Divergenza KL):\n")
            f.write(f"{'Rank':<6} {'Stati':<22} {'Trascrizione Fonemi':<32} {'D_sym':<20}\n")
            f.write("-" * 84 + "\n")
            for rank, (d_val, i, j) in enumerate(most_confusable, 1):
                disp_i = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
                disp_j = get_phoneme_display(j, phonemes, phonemes_ipa, style="full")
                f.write(f"#{rank:<5} Stato {i:02d} - Stato {j:02d}    {disp_i} <-> {disp_j:<28} {d_val:.6f}\n")

            f.write(f"\n2. TOP {top_k} COPPIE CON CUI E' MENO PROBABILE CONFONDERSI (Massima Divergenza KL):\n")
            f.write(f"{'Rank':<6} {'Stati':<22} {'Trascrizione Fonemi':<32} {'D_sym':<20}\n")
            f.write("-" * 84 + "\n")
            for rank, (d_val, i, j) in enumerate(least_confusable, 1):
                disp_i = get_phoneme_display(i, phonemes, phonemes_ipa, style="full")
                disp_j = get_phoneme_display(j, phonemes, phonemes_ipa, style="full")
                f.write(f"#{rank:<5} Stato {i:02d} - Stato {j:02d}    {disp_i} <-> {disp_j:<28} {d_val:.6e}\n")

        # Save complete symmetric KL matrix
        csv_path = os.path.join(output_dir, "kl_divergence_matrix.csv")
        np.savetxt(csv_path, kl_matrix, delimiter=",", fmt="%.6f")

        # 1. Dedicated bar chart for Most and Least Confusable Pairs (Point d)
        plot_confusion_pairs(
            most_confusable=most_confusable,
            least_confusable=least_confusable,
            phonemes=phonemes,
            phonemes_ipa=phonemes_ipa,
            output_dir=output_dir,
            show=show,
            dpi=dpi,
        )

        # 2. MFCC emission feature profiles comparison (physical insight)
        plot_confusion_mfcc_profiles(
            most_confusable=most_confusable,
            least_confusable=least_confusable,
            mean=mean,
            covariance=covariance,
            phonemes=phonemes,
            phonemes_ipa=phonemes_ipa,
            output_dir=output_dir,
            show=show,
            dpi=dpi,
        )

        # 3. Focused confusion submatrix of top involved phonemes
        plot_confusion_submatrix(
            most_confusable=most_confusable,
            kl_matrix=kl_matrix,
            phonemes=phonemes,
            phonemes_ipa=phonemes_ipa,
            output_dir=output_dir,
            show=show,
            dpi=dpi,
        )

        # 4. Global Pairwise Symmetrised KL Divergence Heatmap
        fig, ax = plt.subplots(figsize=(10.5, 9))
        log_kl = np.log10(np.maximum(kl_matrix, 0.0) + 1.0)
        im = ax.imshow(log_kl, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\log_{10}(1 + D_{\mathrm{sym}}(i, j))$")
        ax.set_title("Pairwise Symmetrised KL-Divergence between Phoneme Emissions (Point d)")
        ax.set_xlabel("Phoneme / State Index $j$")
        ax.set_ylabel("Phoneme / State Index $i$")

        if K <= 42:
            step = 2 if K > 20 else 1
            ticks = list(range(0, K, step))
            labels = [get_phoneme_display(t, phonemes, phonemes_ipa, style="short") for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=8)

        # Highlight positions of the most confusable pairs with red markers
        if most_confusable:
            scatter_x = []
            scatter_y = []
            for _, i, j in most_confusable:
                scatter_x.extend([j, i])
                scatter_y.extend([i, j])
            ax.scatter(
                scatter_x,
                scatter_y,
                facecolors="none",
                edgecolors="#ff2222",
                s=80,
                linewidths=1.8,
                marker="o",
                label=f"Top {len(most_confusable)} Most Confusable Pairs",
                zorder=5,
            )
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        kl_png = os.path.join(output_dir, "kl_divergence_heatmap.png")
        kl_pdf = os.path.join(output_dir, "kl_divergence_heatmap.pdf")
        plt.savefig(kl_png, dpi=dpi, bbox_inches="tight")
        plt.savefig(kl_pdf, bbox_inches="tight")
        if show:
            try:
                plt.show()
            except Exception:
                pass
        else:
            plt.close(fig)

        print(f"[OK] Risultati grafici e testuali Punto d salvati con successo in {os.path.abspath(output_dir)}:")
        print(f"     * Report testuale:                 phoneme_confusion_pairs.txt")
        print(f"     * Matrice completa CSV:            kl_divergence_matrix.csv")
        print(f"     * Grafico a barre Most/Least:      phoneme_confusion_pairs.png / .pdf")
        print(f"     * Profili MFCC acustici:           phoneme_confusion_mfcc_profiles.png / .pdf")
        print(f"     * Matrice confusione focalizzata:  phoneme_confusion_submatrix.png / .pdf")
        print(f"     * Heatmap globale annotata:        kl_divergence_heatmap.png / .pdf")

    return kl_matrix


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
    ax1.set_xlabel("EM Iteration (Epoch)")
    ax1.set_ylabel("Total Log-Likelihood")
    ax1.set_title("Log-Likelihood Evolution over Iterations (Point c)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(epochs)

    # Annotate start and end points
    ax1.annotate(
        f"Start: {ll[0]:.2e}",
        xy=(epochs[0], ll[0]),
        xytext=(epochs[0] + 0.2, ll[0]),
        fontsize=9,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    )
    ax1.annotate(
        f"Final: {ll[-1]:.2e}",
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
        ax2.set_xlabel("EM Iteration (Epoch)")
        ax2.set_ylabel(r"$\Delta$ Log-Likelihood ($LL_t - LL_{t-1}$)")
        ax2.set_title("Per-Iteration Improvement (EM Convergence)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.set_xticks(delta_epochs)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)

        # Annotate total improvement
        total_delta = ll[-1] - ll[0]
        ax2.text(
            0.05,
            0.92,
            f"Total improvement:\n+{total_delta:.4e}" if total_delta >= 0 else f"Total change:\n{total_delta:.4e}",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            fontsize=10,
        )
    else:
        ax2.text(0.5, 0.5, "At least 2 epochs required to compute Delta LL", ha="center", va="center")

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


def plot_heatmaps(A, mean, covariance, phonemes=None, phonemes_ipa=None, output_dir="results_plots", show=False, dpi=300):
    """Generate visual heatmaps of Matrix A, Mean vectors, and Covariance variances."""
    if A is None and mean is None and covariance is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Transition Matrix A Heatmap
    if A is not None:
        fig, ax = plt.subplots(figsize=(10, 8.5))
        im = ax.imshow(A, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition Probability $A_{ij}$")
        ax.set_title("Transition Matrix A ($K \\times K$)")
        ax.set_xlabel("Next State $j$")
        ax.set_ylabel("Current State $i$")

        if len(A) <= 42:
            step = 2 if len(A) > 20 else 1
            ticks = list(range(0, len(A), step))
            labels = [get_phoneme_display(t, phonemes, phonemes_ipa, style="short") for t in ticks]
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
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        im = ax.imshow(mean, cmap="coolwarm", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Value $\\mu_{kd}$")
        ax.set_title("Phoneme Emission Means $\\mu$ ($K \\times D$)")
        ax.set_xlabel("MFCC Feature Index ($d = 0 \\dots 39$)")
        ax.set_ylabel("Phoneme / State $k$")

        if len(mean) <= 42:
            step = 2 if len(mean) > 20 else 1
            ticks = list(range(0, len(mean), step))
            labels = [get_phoneme_display(t, phonemes, phonemes_ipa, style="short") for t in ticks]
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
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        im = ax.imshow(np.log10(np.maximum(variances, 1e-12)), cmap="inferno", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\log_{10}(\sigma^2_{kd})$ (Variance)")
        ax.set_title(r"Diagonal Emission Variances ($\log_{10} \mathrm{diag}(\Sigma_k)$)")
        ax.set_xlabel("MFCC Feature Index ($d = 0 \\dots 39$)")
        ax.set_ylabel("Phoneme / State $k$")

        if len(variances) <= 42:
            step = 2 if len(variances) > 20 else 1
            ticks = list(range(0, len(variances), step))
            labels = [get_phoneme_display(t, phonemes, phonemes_ipa, style="short") for t in ticks]
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
    phonemes = data.get("phonemes")
    phonemes_ipa = data.get("phonemes_ipa")

    # Output directory
    output_dir = args.output_dir

    # 1. Print Matrices Report
    print_matrices_report(
        mean=mean,
        covariance=covariance,
        A=A,
        phonemes=phonemes,
        phonemes_ipa=phonemes_ipa,
        full=args.full,
        specific_state=args.state,
        output_dir=output_dir,
    )

    # 2. Sanity Check: Phoneme Confusion Pairs via Symmetrised KL Divergence (Point d / Exercise 2.4)
    compute_and_print_kl_confusion(
        mean=mean,
        covariance=covariance,
        phonemes=phonemes,
        phonemes_ipa=phonemes_ipa,
        top_k=args.top_k,
        include_boundary=args.include_boundary,
        output_dir=output_dir,
        show=args.show,
        dpi=args.dpi,
    )

    # 3. Plot Log-Likelihood Evolution (Point c / Point 3)
    has_display = bool(os.environ.get("DISPLAY"))
    should_show = args.show or (has_display and not args.no_show and not args.show)

    plot_ll_evolution(ll=ll, output_dir=output_dir, show=args.show, dpi=args.dpi)

    # 4. Generate Visual Heatmaps of the Matrices
    print("\n[INFO] Generazione heatmap visive per le matrici...")
    plot_heatmaps(
        A=A,
        mean=mean,
        covariance=covariance,
        phonemes=phonemes,
        phonemes_ipa=phonemes_ipa,
        output_dir=output_dir,
        show=args.show,
        dpi=args.dpi,
    )

    print("\nElaborazione completata con successo!\n")


if __name__ == "__main__":
    main()
