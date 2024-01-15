import skimage.measure
import numpy as np
import db_util
import openl3
import tqdm
import soundfile as sf
import torch
import kornia
import pandas as pd

EMBEDDING_SIZE = 512
BATCH_SIZE = 8

def max_pool(embeddings):
	return skimage.measure.block_reduce(embeddings, (EMBEDDING_SIZE, 1), np.max)[0]

def make_batches(dataframe, batch_size):
    batches = []
    # append batches of rows to batches as dictionaries
    for i in range(0, len(dataframe), batch_size):
        batch = dataframe[i:i + batch_size]
        batches.append(batch.to_dict('records'))
    return batches

def embed_tracks(tracks, sub_song_length=15, overwrite=False):
    unembedded_tracks = []
    for i, track in tracks.iterrows():
        if len(track.embedding) == 0:
            unembedded_tracks.append(track)
    tracks = unembedded_tracks
    tracks = pd.DataFrame(tracks)
    batches = make_batches(tracks, BATCH_SIZE)
    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=EMBEDDING_SIZE)
    outputs = np.zeros((len(tracks), EMBEDDING_SIZE))
    for batch in tqdm.tqdm(batches, total=len(tracks) // BATCH_SIZE):
        sound_files = [sf.read(track['audio_path']) for track in batch]
        audios = [sound_file[0] for sound_file in sound_files]
        sample_rates = [sound_file[1] for sound_file in sound_files]
        for i in range(len(audios)):
            # get center sub-song of length sub_song_length
            samples_per_thirty_seconds = sample_rates[i] * sub_song_length
            middle = len(audios[i]) // 2
            audios[i] = audios[i][middle - samples_per_thirty_seconds // 2:middle + samples_per_thirty_seconds // 2]
        emb_list, _ = openl3.get_audio_embedding(audios, sample_rates, batch_size=BATCH_SIZE, model=model)
        max_pooled = [max_pool(embeddings) for embeddings in emb_list]
        # save embeddings to database
        db_util.add_embeddings([track['id'] for track in batch], max_pooled)

class SpacingLoss(torch.nn.Module):
    def __init__(self, original, precision=100):
        super().__init__()
        self.original = original
        self.feature_orders = [torch.argsort(self.original[:,i]) for i in range(self.original.shape[1])]
        self.entropy_weight = 1
        self.order_weight = 2
        self.precision = precision
        self.bandwidth = 0.5

    def forward(self, x, return_separate_losses=False):
        feature_entropies = torch.zeros(x.shape[1], device=x.device)
        n_bins = self.precision
        for j in range(x.shape[1]):
            data_tensor = x[:,j]
            
            # Estimate the probability density function (PDF) using histogram
            bins = torch.torch.linspace(data_tensor.min().item(), data_tensor.max().item(), n_bins, device=x.device)
            hist = kornia.enhance.histogram(data_tensor.unsqueeze(0), bins, torch.tensor(0.5), epsilon=1e-10).to(x.device)
            
            # Calculate the bin widths
            bin_widths = 1 / n_bins

            # Normalize histogram to get probabilities
            probabilities = hist * bin_widths

            # Calculate entropy using the formula: -sum(p(x) * log(p(x)))
            probabilities = probabilities + 0.01
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            feature_entropies[j] = entropy

        entropy_loss = -torch.mean(feature_entropies)
        order_errors = torch.zeros(x.shape[1], device=x.device)
        for i in range(x.shape[1]):
            feature = x[:,i]
            current_feature_order = torch.argsort(feature)
            original_feature_order = self.feature_orders[i]
            mae = torch.mean(torch.abs(current_feature_order - original_feature_order).type(torch.float32))
            order_errors[i] = mae
        order_loss = torch.mean(order_errors)
        total_loss = self.entropy_weight * entropy_loss + self.order_weight * order_loss
        if return_separate_losses:
            return total_loss, entropy_loss, order_loss
        return total_loss
    
def optimize_spacing(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32, device=device)
    criterion = SpacingLoss(x).to(device)
    optimizer = torch.optim.Adam([x], lr=0.1)

    print("Optimizing embedding spacing...")
    for i in tqdm.tqdm(range(100)):
        optimizer.zero_grad()
        total_loss, entropy_loss, order_loss = criterion(x, return_separate_losses=True)
        total_loss.backward()
        optimizer.step()
    return x.detach().cpu().numpy()