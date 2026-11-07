#!/bin/bash

#SBATCH --job-name=viterbi_hmm        # Nome del job
#SBATCH --partition=GPU-a40          # Partizione GPU da usare
#SBATCH --nodes=1                    # Numero di nodi
#SBATCH --ntasks=1                   # Numero di task
#SBATCH --cpus-per-task=4            # Core CPU riservati
#SBATCH --gres=gpu:a40:1             # 1 GPU NVIDIA A40
#SBATCH --mem=16G                    # Memoria RAM di sistema richiesta
#SBATCH --time=02:00:00              # Tempo massimo di esecuzione (HH:MM:SS)
#SBATCH --output=viterbi_%j.log      # File di log (%j inserisce il JobID)

echo "=========================================================="
echo "Inizio esecuzione Viterbi HMM (Punto e / Esercizio 2.4)"
echo "Data: $(date)"
echo "Nodo assegnato: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================================="
nvidia-smi -L

# 1. Carica l'ambiente Python (micromamba)
eval "$(micromamba shell hook --shell bash 2>/dev/null || ~/.local/bin/micromamba shell hook --shell bash 2>/dev/null)"
micromamba activate mllab || source ~/.bashrc && micromamba activate mllab

# Assicurati che Levenshtein sia installato
python -c "import Levenshtein" 2>/dev/null || pip install Levenshtein 2>/dev/null || true

# 2. Cartella del codice (dove si trova questo script al momento del submit)
code_dir="$SLURM_SUBMIT_DIR"

# 3. Cartella persistente su /share
final_dir="/share/${USER}/project1"
mkdir -p "${final_dir}"

# 4. Prova a usare /scratch locale ad alte prestazioni se disponibile
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    work_dir="/scratch/${USER}_${SLURM_JOB_ID}"
    mkdir -p "${work_dir}"
    chmod 700 "${work_dir}"
    export RESULTS_DIR="${work_dir}"
    USE_SCRATCH=1
    echo "Utilizzo scratch locale ad alta velocità: ${work_dir}"
else
    work_dir="${final_dir}"
    export RESULTS_DIR="${final_dir}"
    USE_SCRATCH=0
    echo "Utilizzo cartella condivisa persistente: ${work_dir}"
fi

cd "${work_dir}"

# 5. Recupero del file trained_params.npz e collegamento di plotting_functions
if [ "${final_dir}" != "${work_dir}" ] && [ -f "${final_dir}/trained_params.npz" ]; then
    echo "Copia di trained_params.npz da ${final_dir} in ${work_dir}..."
    cp "${final_dir}/trained_params.npz" "${work_dir}/"
elif [ "${code_dir}" != "${work_dir}" ] && [ -f "${code_dir}/trained_params.npz" ]; then
    echo "Copia di trained_params.npz da ${code_dir} in ${work_dir}..."
    cp "${code_dir}/trained_params.npz" "${work_dir}/"
fi

# Rende accessibile la cartella plotting_functions/ anche da dentro work_dir
if [ ! -e "${work_dir}/plotting_functions" ] && [ -d "${code_dir}/plotting_functions" ]; then
    ln -s "${code_dir}/plotting_functions" "${work_dir}/plotting_functions" 2>/dev/null || true
fi

# 6. Variabili d'ambiente per Python
export PYTHONPATH="${code_dir}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# 7. Esecuzione della decodifica Viterbi (Punto e)
echo "Avvio di viterbi_hmm.py..."
python -u "${code_dir}/viterbi_hmm.py"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "=========================================================="
    echo "[ERRORE] viterbi_hmm.py è fallito con codice di uscita: $EXIT_CODE"
    echo "Controlla il traceback sopra per i dettagli dell'errore."
    echo "=========================================================="
    exit $EXIT_CODE
fi

# 8. Generazione automatica dei grafici degli errori (istogrammi punto e)
if [ -f "${work_dir}/viterbi_results.npz" ]; then
    echo "Generazione grafici e istogrammi con plot_viterbi_results.py..."
    python -u "${code_dir}/plotting_functions/plot_viterbi_results.py" \
        "${work_dir}/viterbi_results.npz" \
        --output-dir "${work_dir}/results_plots"
fi

# 9. Copia dei risultati nella cartella persistente su /share e pulizia scratch
if [ "$USE_SCRATCH" -eq 1 ]; then
    echo "Copia dei risultati finali da ${work_dir} a ${final_dir}..."
    cp -r "${work_dir}"/* "${final_dir}/" 2>/dev/null
    rm -rf "${work_dir}"
fi

echo "=========================================================="
echo "Job completato con successo: $(date)"
echo "Tutti i risultati e i grafici sono salvati in: ${final_dir}"
echo "=========================================================="
