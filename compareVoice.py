from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from tqdm import tqdm
import numpy as np

def compute_similarity():

## Load and preprocess the audio
    data_dir = Path("voices")
    wav_fpaths = list(data_dir.glob("**/*.wav"))
    wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]


## Compute the embeddings
    encoder = VoiceEncoder()
    embeds = np.array([encoder.embed_utterance(wav) for wav in wavs])
    speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
    names = np.array([fpath.stem for fpath in wav_fpaths])


    # Take 6 real embeddings at random, and leave the 6 others for testing
    gt_indices = np.random.choice(*np.where(speakers == "real"), 1, replace=False) 
    mask = np.zeros(len(embeds), dtype=bool)
    mask[gt_indices] = True
    gt_embeds = embeds[mask]
    gt_names = names[mask]
    gt_speakers = speakers[mask]
    embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]


    ## Compare all embeddings against the ground truth embeddings, and compute the average similarities.
    scores = (gt_embeds @ embeds.T).mean(axis=0)

    # Order the scores by decreasing order
    sort = np.argsort(scores)[::-1]
    scores, names, speakers = scores[sort], names[sort], speakers[sort]

    return scores[0]  # Returning the top similarity score


similarity_score = compute_similarity()
print(f"Similarity score: {similarity_score}")