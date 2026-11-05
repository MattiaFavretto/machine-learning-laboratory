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

# Cartella di lavoro temporanea e cartella persistente per i risultati
# Usa /scratch se disponibile e scrivibile; se siamo su un nodo virtuale (es. i-*) usa /tmp come fallback
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    scratch_dir="/scratch/${USER}_${SLURM_JOB_ID}"
else
    scratch_dir="/tmp/${USER}_${SLURM_JOB_ID}"
fi

mkdir -p "${scratch_dir}"
chmod 700 "${scratch_dir}"

export FINAL_RESULTS_DIR="/share/${USER}/project1"
mkdir -p "${FINAL_RESULTS_DIR}"

# Durante il job scrivi su scratch/tmp per non saturare la rete/I/O del cluster
export RESULTS_DIR="${scratch_dir}"

cd "${scratch_dir}"

# Permette a Python di trovare data_loader_speech.py e generate_chains.py
export PYTHONPATH="${code_dir}:${PYTHONPATH}"

# Esecuzione dello script Python 
python "${code_dir}/project_1.py"

# Copia dei risultati dalla scratch alla cartella persistente su /share
cp -r ${scratch_dir}/* "${FINAL_RESULTS_DIR}/" 2>/dev/null
echo "Risultati salvati in ${FINAL_RESULTS_DIR}"

# Pulizia della cartella scratch temporanea come da cluster etiquette
rm -rf "${scratch_dir}"
