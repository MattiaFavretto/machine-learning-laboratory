### START CLAUDE ##############################################################
### PUNTO e) - PARTE AGGIUNTA - Viterbi sul parlato, Levenshtein, istogramma ###
###############################################################################
# Perche' non si puo' usare direttamente h_best_viterbi_log sui dati reali:

#  2. e' compilata per ogni lunghezza T diversa (ricompilazione per ogni frase).
#     Qui tutte le frasi sono padded a MAX_T e i frame oltre T_len sono mascherati.
#  3. non sa che il modello e' left-to-right: START all'inizio, END alla fine.
#  4. lavora su una frase alla volta: qui e' vettorizzata (vmap) su un batch.

MAX_T = 199          # data_iter() scarta le frasi con >= 200 frame
N_TRAIN = 700        # punto c): le prime 700 frasi sono il training set
H_BEST = 30          # punto e): 30 trascrizioni piu' probabili
START_STATE = 40     # indice dello stato START in dl.get_phonemes()
END_STATE = 41       # indice dello stato END

def _h_best_viterbi_masked(log_emissions, T_len, log_trans, log_init, final_state, h):
    """h-best Viterbi in log-space su UNA frase con emissioni precalcolate e padding.

    log_emissions: (MAX_T, K) | T_len: lunghezza reale | final_state: stato in cui
    la catena DEVE finire (-1 = libero). Ritorna (h, MAX_T) sequenze e (h,) LL.
    Per t >= T_len il passo e' l'identita' (carry invariato, backpointer = se stesso)."""
    k = log_trans.shape[0]
    v0 = log_init + log_emissions[0]
    prev_step = jnp.full((k, h), -jnp.inf).at[:, 0].set(v0)
    identity_j = jnp.broadcast_to(jnp.arange(k)[:, None], (k, h))
    identity_r = jnp.broadcast_to(jnp.arange(h)[None, :], (k, h))

    def forward(carry, inp):
        em_t, t = inp
        transition = carry[:, :, None] + log_trans[:, None, :]            # (k, h, k)
        top_val, top_idx = lax.top_k(transition.reshape(k * h, k).T, h)  # (k, h)
        new_carry = top_val + em_t[:, None]
        valid = t < T_len
        new_carry = jnp.where(valid, new_carry, carry)
        best_j = jnp.where(valid, top_idx // h, identity_j)
        best_r = jnp.where(valid, top_idx % h, identity_r)
        return new_carry, (best_j, best_r)

    t_idx = jnp.arange(1, log_emissions.shape[0])
    V, (B_j, B_r) = lax.scan(forward, prev_step, (log_emissions[1:], t_idx))

    # la catena deve terminare nello stato END (se final_state >= 0)
    final_mask = jnp.where((final_state < 0) | (jnp.arange(k) == final_state), 0.0, -jnp.inf)
    best_log_probs, flat = lax.top_k((V + final_mask[:, None]).ravel(), h)
    last_states, last_ranks = flat // h, flat % h

    def backward(carry, b_t):
        states, ranks = carry
        b_j_t, b_r_t = b_t
        prev_states = b_j_t[states, ranks]
        prev_ranks = b_r_t[states, ranks]
        return (prev_states, prev_ranks), prev_states

    _, map_seq = lax.scan(backward, (last_states, last_ranks), (B_j, B_r), reverse=True)
    return jnp.vstack([map_seq, last_states[None, :]]).T, best_log_probs


def make_batched_decoder(h=H_BEST):
    """Decoder jit+vmap su un batch di frasi: (B,MAX_T,K), (B,) -> (B,h,MAX_T), (B,h)."""
    core = partial(_h_best_viterbi_masked, h=h)
    return jit(vmap(core, in_axes=(0, 0, None, None, None)))


def collapse_duplicates(path):
    """Elimina i ripetuti consecutivi
    (stato mantenuto per piu' frame -> un solo fonema) => trascrizione."""
    path = np.asarray(path)
    if len(path) == 0:
        return path
    mask = np.ones(len(path), dtype=bool)
    mask[1:] = path[1:] != path[:-1]
    return path[mask]


def levenshtein(a, b):
    """Distanza di Levenshtein (inserimenti, cancellazioni, sostituzioni costo 1).
    Righe della DP vettorizzate: new[j] = min_{k<=j}(tmp[k] + j - k) = j + cummin(tmp - j)."""
    a, b = np.asarray(a), np.asarray(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    j = np.arange(m + 1)
    prev = j.copy()
    for i in range(1, n + 1):
        tmp = np.empty(m + 1, dtype=np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != a[i - 1]))
        prev = np.minimum.accumulate(tmp - j) + j
    return int(prev[m])


def weighted_relative_error(true_transcription, paths, lls, cache=None):
    """Punto e): Levenshtein di ognuna delle h trascrizioni (dopo collapse_duplicates) rispetto
    al ground truth, mediate con pesi softmax(LL), divise per la lunghezza vera."""
    finite = np.isfinite(lls)
    if not finite.any():
        return np.nan
    weights = softmax(np.where(finite, lls, -np.inf))
    cache = {} if cache is None else cache
    total = 0.0
    for p, w in zip(paths, weights):
        if w == 0.0:
            continue
        trans = tuple(collapse_duplicates(p).tolist())
        if trans not in cache:
            cache[trans] = levenshtein(trans, true_transcription)
        total += w * cache[trans]
    return total / len(true_transcription)


def decode_test_set(mean, covariance, A, h=H_BEST, batch_size=64, max_test=None, n_train=N_TRAIN):
    """Viterbi h-best su tutte le frasi DOPO le prime n_train (= il test set del punto e)).
    Ritorna: errori relativi, lunghezze T, trascrizioni vere, esempi (vero, decodificato, LL)."""
    from itertools import islice
    K = mean.shape[0]
    mean_j = jnp.asarray(mean, dtype=jnp.float32)
    cov_j = jnp.asarray(covariance, dtype=jnp.float32)
    with np.errstate(divide="ignore"):                       # A ha zeri esatti -> log = -inf (corretto)
        log_trans = jnp.log(jnp.asarray(A, dtype=jnp.float32))
        # il training parte sempre da START (pi = [1,0,..]): stessa assunzione nel decoding
        log_init = jnp.log(jnp.zeros(K).at[START_STATE].set(1.0))
    decoder = make_batched_decoder(h)

    test_iter = islice(dl.data_iter(), n_train, None if max_test is None else n_train + max_test)
    errors, lengths, truths, examples, cache = [], [], [], [], {}
    skipped = 0

    def process(batch):
        nonlocal skipped
        X = np.zeros((len(batch), MAX_T, mean.shape[1]), dtype=np.float32)
        T_lens = np.array([len(x) for x, _ in batch], dtype=np.int32)
        for i, (x, _) in enumerate(batch):
            X[i, :len(x)] = x
        pdf = p1.logpdf_vmap(jnp.asarray(X), mean_j, cov_j)
        seqs, lls = decoder(pdf, jnp.asarray(T_lens), log_trans, log_init, END_STATE)
        seqs, lls = np.asarray(seqs), np.asarray(lls)
        for i, (x, true_trans) in enumerate(batch):
            paths = [seqs[i, r, :T_lens[i]] for r in range(h)]
            err = weighted_relative_error(tuple(np.asarray(true_trans).tolist()), paths, lls[i], cache)
            if np.isnan(err):
                skipped += 1      # nessun cammino valido (es. T < numero di fonemi)
                continue
            errors.append(err)
            lengths.append(int(T_lens[i]))
            truths.append(tuple(np.asarray(true_trans).tolist()))
            if len(examples) < 3:
                examples.append((np.asarray(true_trans), collapse_duplicates(paths[0]), float(lls[i, 0])))

    batch = []
    for n, item in enumerate(test_iter, 1):
        batch.append(item)
        if len(batch) == batch_size:
            process(batch)
            batch = []
            print(f"  decodificate {n} frasi di test...", flush=True)
    if batch:
        process(batch)
    if skipped:
        print(f"[ATTENZIONE] {skipped} frasi scartate: nessun cammino valido START->END")
    return errors, lengths, truths, examples


def random_baseline_errors(mean, covariance, A, lengths, true_transcriptions, n_samples=200, seed=0):
    """Baseline del testo: sequenze CASUALI generate dal Markov chain sampler (generate_chains.py)
    con i parametri STIMATI (mean, covariance, A), confrontate con il ground truth.
    Per ogni frase di test scelta si genera una catena della stessa lunghezza (in frame); si
    raggruppa per lunghezza perche' generate_chain ricompila per ogni valore di `length`.
    Ritorna (indici delle frasi usate, errori relativi)."""
    K, D = mean.shape
    A64 = np.asarray(A, dtype=np.float64)
    A64 = A64 / A64.sum(axis=1, keepdims=True)
    init = np.zeros(K)
    init[START_STATE] = 1.0                       # come nel training: si parte da START
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(lengths), size=min(n_samples, len(lengths)), replace=False))

    by_length = {}
    for i in idx:
        by_length.setdefault(int(lengths[i]), []).append(int(i))

    errs = {}
    for L, group in by_length.items():
        chains = generate_chains(K, len(group), length=L, d=D,
                                 μ=np.asarray(mean, dtype=np.float64),
                                 Σ=np.asarray(covariance, dtype=np.float64),
                                 transition_probs=A64, initial_probs=init)
        for i, chain in zip(group, chains):
            random_trans = np.asarray(chain[2])   # 3o valore: stati senza duplicati = "trascrizione"
            true_trans = true_transcriptions[i]
            errs[i] = levenshtein(random_trans, true_trans) / len(true_trans)
    return idx, np.array([errs[int(i)] for i in idx])


def main_point_e(args):
    """Punto e) di Exercise 2.4: (1) Viterbi 30-best + errore relativo, (2) baseline casuale con
    generate_chains, (3) export di tutti i dati in un file .npz (la stampa e' fatta da
    plot_viterbi_results.py)."""
    params = np.load(args.params)
    mean, covariance, A = params["mean"], params["covariance"], params["A"]
    print(f"Parametri caricati da {args.params}: mean{mean.shape} cov{covariance.shape} A{A.shape}")

    # ---------------- (1) Viterbi h-best sul test set ----------------
    print(f"Viterbi {args.h}-best sul test set (frasi dopo le prime {args.n_train})...")
    start = timeit.default_timer()
    errors, lengths, truths, examples = decode_test_set(mean, covariance, A, h=args.h, batch_size=args.batch_size,
                                                        max_test=args.max_test, n_train=args.n_train)
    print(f"Tempo di decoding: {timeit.default_timer() - start:.1f} s su {len(errors)} frasi")
    errors = np.array(errors)

    names = list(params["phonemes"]) if "phonemes" in params.files else None
    if names is None:
        try:
            _, names = dl.get_phonemes()
        except Exception:
            names = None
    fmt = (lambda s: " ".join(names[i] for i in s)) if names is not None else (lambda s: " ".join(map(str, s)))
    print("\nEsempi (vero vs miglior cammino Viterbi):")
    for true_t, dec_t, ll in examples:
        print(f"  VERO   : {fmt(true_t)}\n  VITERBI: {fmt(dec_t)}   (LL = {ll:.1f})\n")

    mean_err = float(np.mean(errors))
    print("=" * 70)
    print(f"ERRORE RELATIVO MEDIO (Viterbi, softmax-pesato sulle {args.h} migliori): {mean_err:.4f} = {100*mean_err:.1f}%")
    print("=" * 70)

    # ---------------- (2) baseline: sequenze casuali con generate_chains ----------------
    baseline_idx, baseline_errors = np.array([], dtype=int), np.array([])
    if args.n_random > 0:
        print(f"Baseline: {min(args.n_random, len(lengths))} sequenze casuali da generate_chains (parametri stimati)...")
        baseline_idx, baseline_errors = random_baseline_errors(mean, covariance, A, lengths, truths,
                                                               n_samples=args.n_random)
        print(f"ERRORE RELATIVO MEDIO (sequenze casuali): {baseline_errors.mean():.4f} = {100*baseline_errors.mean():.1f}%")

    # ---------------- (3) export ----------------
    np.savez(
        args.out,
        viterbi_errors=errors,                                   # errore relativo per ogni frase di test
        viterbi_mean_error=mean_err,
        test_frames=np.array(lengths),                           # T (frame) di ogni frase di test
        test_true_length=np.array([len(t) for t in truths]),     # lunghezza della trascrizione vera
        random_errors=baseline_errors,                           # errore relativo delle sequenze casuali
        random_mean_error=float(baseline_errors.mean()) if len(baseline_errors) else np.nan,
        random_utterance_idx=baseline_idx,                       # a quali frasi di test corrispondono
        h=args.h,
        n_train=args.n_train,
    )
    print(f"Risultati esportati in {args.out}  ->  stampa con: python plot_viterbi_results.py {args.out}")
### END CLAUDE ################################################################


if __name__ == "__main__":
    ### START CLAUDE ##########################################################
    parser = argparse.ArgumentParser(description="Punto e) di Exercise 2.4: Viterbi h-best sul test set del parlato")
    parser.add_argument("--params", default="trained_params.npz", help="npz di project_1.py (mean, covariance, A)")
    parser.add_argument("--h", type=int, default=H_BEST, help="numero di trascrizioni migliori (default 30)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="frasi usate nel training, escluse dal test (700)")
    parser.add_argument("--max-test", type=int, default=None, help="limita il numero di frasi di test (debug)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-random", type=int, default=200, help="frasi per la baseline casuale (0 = salta)")
    parser.add_argument("--out", default="viterbi_results.npz", help="file .npz con tutti i risultati esportati")
    main_point_e(parser.parse_args())
    ### END CLAUDE ############################################################