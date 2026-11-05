#!/bin/bash

#SBATCH --job-name=project1          # Nome del job
#SBATCH --partition=GPU-a40          # Partizione GPU da usare
#SBATCH --nodes=1                    # Numero di nodi
#SBATCH --ntasks=1                   # Numero di task
#SBATCH --cpus-per-task=4            # Core CPU riservati
#SBATCH --gres=gpu:a40:1             # 1 GPU NVIDIA A40
#SBATCH --mem=16G                    # Memoria RAM di sistema richiesta
#SBATCH --time=02:00:00              # Tempo massimo di esecuzione (HH:MM:SS)
#SBATCH --output=project1_%j.log     # File di log (%j inserisce il JobID)

# Informazioni sul nodo assegnato
echo "Esecuzione sul nodo: $SLURM_JOB_NODELIST"
nvidia-smi -L

# Carica l'ambiente Python (micromamba)
eval "$(micromamba shell hook --shell bash 2>/dev/null || ~/.local/bin/micromamba shell hook --shell bash 2>/dev/null)"
micromamba activate mllab || source ~/.bashrc && micromamba activate mllab

# Cartella del codice (dove si trova questo script al momento del submit)
code_dir="$SLURM_SUBMIT_DIR"

# Cartella persistente su /share (sempre montata e scrivibile su tutti i nodi)
final_dir="/share/${USER}/project1"
mkdir -p "${final_dir}"

# Prova a usare /scratch locale se disponibile e scrivibile; altrimenti scrivi direttamente su /share
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    work_dir="/scratch/${USER}_${SLURM_JOB_ID}"
    mkdir -p "${work_dir}"
    chmod 700 "${work_dir}"
    export RESULTS_DIR="${work_dir}"
    USE_SCRATCH=1
else
    work_dir="${final_dir}"
    export RESULTS_DIR="${final_dir}"
    USE_SCRATCH=0
fi

cd "${work_dir}"

# Permette a Python di trovare data_loader_speech.py e generate_chains.py
export PYTHONPATH="${code_dir}:${PYTHONPATH}"

# Esecuzione dello script Python 
python "${code_dir}/project_1.py"

# Se è stata usata la scratch temporanea locale, copia su /share e ripulisci
if [ "$USE_SCRATCH" -eq 1 ]; then
    cp -r "${work_dir}"/* "${final_dir}/" 2>/dev/null
    rm -rf "${work_dir}"
fi

echo "Risultati salvati in ${final_dir}"
