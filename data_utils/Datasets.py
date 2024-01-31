from .BinaryTokenizer import BinaryTokenizer
from torch.utils.data import Dataset
import numpy as np

class TokenizedChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        melody_pcps = data['melody_pcps']
        # pad to start with zero
        chord_pcps = np.pad(data['chord_pcps'], ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)
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
        self.chord_pcps = np.pad(data['chord_pcps'].astype('float32'), ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        return self.melody_pcps[idx,:,:], self.chord_pcps[idx,:,:]
    # end __getitem__
# end BinChromaDataset

class PermutationsTokenizedChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps']
        self.chord_pcps = np.pad(data['chord_pcps'], ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)
        self.binTok = BinaryTokenizer()
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        for i in range(m.shape[1]):
            p = np.random.permutation(12)
            m[i,:] = m[i,p]
            c[i,:] = c[i,p]
        self.binTok.fit_transform( c )
        self.binTok.fit_transform( m )
        return self.binTok.fit_transform( m ), self.binTok.fit_transform( c )
    # end __getitem__
# end TokenizedChromaDataset

class PermutationsBinChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = np.pad(data['chord_pcps'].astype('float32'), ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        for i in range(m.shape[1]):
            p = np.random.permutation(12)
            m[i,:] = m[i,p]
            c[i,:] = c[i,p]
        return m, c
    # end __getitem__
# end BinChromaDataset