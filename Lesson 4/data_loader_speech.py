from scipy import sparse, fftpack
from scipy.io import wavfile
import numpy as np

## helpers for calculating MFCC
window = np.hamming(550)
frequencies = np.fft.fftfreq(550, 1/22050)[:225]
mels = 1127 * np.log(frequencies/700 + 1)
division = mels[-1]/39
indices = [np.searchsorted(mels, i*division-1e-5) for i in range(0,40)]
values = []
ind = []
for i, (start, middle, end) in enumerate(zip([0,] + indices[:-1], indices, indices[1:] + [indices[-1],])):
    slope1 = middle-start
    values += [x/slope1 for x in range(1,slope1)]
    ind += [(i,k) for k in range(start+1, middle)]
    slope2 = end-middle
    values += [1,] + [(x-1)/slope2 for x in range(slope2,1,-1)]
    ind += [(i,middle)] + [(i,k) for k in range(middle+1, end)]


to_mel = sparse.coo_array((np.array(values), np.array(ind).T), shape=(40,225)).T.tocsr()

def get_sentence(Id):
    samplerate, data = wavfile.read(f'/share/datasets/LJSpeech-1.1/wavs/{Id}.wav')
    assert samplerate == 22050
    d = np.lib.stride_tricks.sliding_window_view(data, 550)[::330, :] # Only a view, no reallocation
    data_windowed = d * window
    f = np.fft.fft(data_windowed, axis=1)[:, :225]
    fl = np.log(np.abs(f))
    m = fl @ to_mel
    return fftpack.dct(m)

# def get_sentence_lazy(Id):
#     samplerate, data = wavfile.read(f'/share/datasets/LJSpeech-1.1/wavs/{Id}.wav')
#     assert samplerate == 22050
#     return len(data)//330


def read_csv(fname, sep=","):
    with open(fname, "r") as f:
        return [s.strip().split(sep) for s in f]

def get_phonemes():
    phoneme_list = [a[0] for a in read_csv('/share/datasets/LJSpeech-1.1/sphinxdict/SphinxPhones_40')]
    phoneme_list.append("START")
    phoneme_list.append("END")
    phonemes = {phoneme : i for i, phoneme in enumerate(phoneme_list)}
    return phonemes, phoneme_list

def get_phonemes2ipa():
    translation = {a[0] : a[1] for a in read_csv('/share/datasets/LJSpeech-1.1/sphinxdict/toIPA.csv')}
    translation["SIL"] = " "
    translation["START"] = ""
    translation["END"] = ""
    return lambda x: "".join(map(translation.get, x))

# the fuction that returns the sentences and the transcription
def data_iter():
    phonemes, _ = get_phonemes()
    transcriptions = read_csv("/share/datasets/LJSpeech-1.1/transcripts.txt", "|")
    for i, (Id, trans) in enumerate(transcriptions):
        sentence = get_sentence(Id)
        if len(sentence) < 200:
            yield (sentence, np.array([phonemes[ph] for ph in trans.split(" ")]))
            # returns a function that needs to be called to have the results
