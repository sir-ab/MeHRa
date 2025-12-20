import librosa
import torch
import torchaudio
from kokoro.model import KModel
from huggingface_hub import hf_hub_download

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = hf_hub_download(repo_id="SirAB/styletts2_ljspeech_finetune", filename="epoch_2nd_00004.pth")





model = torch.load("styletts2_ljspeech_finetune.pth", map_location='cpu', weights_only=True)
# state = model['net']
# torch.save(state, 'styletts2_ljspeech_finetune.pth')

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor



def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)