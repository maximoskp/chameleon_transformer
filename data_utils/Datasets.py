from .BinaryTokenizer import BinaryTokenizer
from torch.utils.data import Dataset
import numpy as np

class TokenizedChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        melody_pcps = data['melody_pcps']
        chord_pcps = data['chord_pcps']
        binTok = BinaryTokenizer()
        self.tok_melody = binTok.fit_transform( melody_pcps )
        self.tok_chord = binTok.fit_transform( chord_pcps )
    # end __init__
    
    def __len__(self):
        return self.tok_melody.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        return self.tok_melody[idx,:], self.tok_chord[idx,:]
    # end __getitem__
# end TokenizedChromaDataset

class BinChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        return self.melody_pcps[idx,:,:], self.chord_pcps[idx,:,:]
    # end __getitem__
# end BinChromaDataset