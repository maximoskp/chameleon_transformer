from .BinaryTokenizer import BinaryTokenizer
from torch.utils.data import Dataset
import numpy as np

class TokenizedConcatChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        melody_pcps = data['melody_pcps']
        chord_pcps = np.pad(data['chord_pcps'], ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)
        binTok = BinaryTokenizer()
        tok_melody = binTok.fit_transform( melody_pcps )
        self.tok_sequence = np.concatenate( (tok_melody , binTok.fit_transform( chord_pcps )), axis=1 )
    # end __init__
    
    def __len__(self):
        return self.tok_sequence.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        return self.tok_sequence[idx,:]
    # end __getitem__
# end TokenizedConcatChromaDataset

class BinConcatChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        melody_pcps = data['melody_pcps'].astype('float32')
        self.sequence = np.concatenate( (melody_pcps, np.pad(data['chord_pcps'].astype('float32'), ( (0,0), (1,0), (0,0) ), mode='constant', constant_values=1)), axis=1 )
    # end __init__
    
    def __len__(self):
        return self.sequence.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        return self.sequence[idx,:,:]
    # end __getitem__
# end BinConcatChromaDataset

class PermTokenizedConcatChromaDataset(Dataset):
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
        return np.concatenate( (self.binTok.fit_transform( m ) , self.binTok.fit_transform( c )) )
    # end __getitem__
# end PermTokenizedConcatChromaDataset

class PermBinConcatChromaDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        # padding in chord_pcps needs to be performed after permutation, not here
        self.chord_pcps = data['chord_pcps'].astype('float32')
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
        return np.concatenate( (m, np.pad(c, ( (1,0), (0,0) ), mode='constant', constant_values=1)), axis=0 )
    # end __getitem__
# end PermBinConcatChromaDataset