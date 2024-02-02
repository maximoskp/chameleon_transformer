from .BinaryTokenizer import BinaryTokenizer
from torch.utils.data import Dataset
import numpy as np

class BinarySerializer:
    def __init__(self, pad_to_length=0):
        '''
        0: padding
        1: start_melody
        2: melody segment seperator
        3: chord segment separator
        4: start harmonizing
        5: end harmonizing
        6 to 6+11=17: melody pitch class
        18 to 18+11=29: chord pitch class
        
        vocab_size = 30
        '''
        self.padding = 0
        self.start_melody = 1
        self.melody_segment_separator = 2
        self.chord_segment_separator = 3
        self.start_harmonizing = 4
        self.end_harmonizing = 5
        self.melody_offset = 6
        self.chord_offset = 18
        self.max_seq_length = 0
        self.pad_to_length = pad_to_length
        self.vocab_size = 30
    # end init
    def sequence_serialization(self, melody, chords):
        t = []
        t.append(self.start_melody)
        for i in range(melody.shape[0]):
            # check if melody pcs exist
            m = melody[i,:]
            nzm = np.nonzero(m)[0]
            if len(nzm) > 0:
                t.append( self.melody_segment_separator )
                t.extend( nzm + self.melody_offset )
        t.append(self.start_harmonizing)
        for i in range(chords.shape[0]):
            # check if chord pcs exist
            c = chords[i,:]
            nzc = np.nonzero(c)[0]
            if len(nzc) > 0:
                t.append( self.chord_segment_separator )
                t.extend( nzc + self.chord_offset )
        t.append(self.end_harmonizing)
        if len(t) > self.max_seq_length:
            self.max_seq_length = len(t)
        while self.pad_to_length > len(t):
            t.append(self.padding)
        return np.array(t)
    # end sequence_serialization
# end class BinarySerializer

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

class PermSerializedConcatDataset(Dataset):
    def __init__(self, npz_path, pad_to_length=1100):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.binser = BinarySerializer(pad_to_length=pad_to_length)
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
        t = self.binser.sequence_serialization( m , c )
        return t
    # end __getitem__
# end PermSerializedConcatDataset

class SerializedConcatDataset(Dataset):
    def __init__(self, npz_path, pad_to_length=1100):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.binser = BinarySerializer(pad_to_length=pad_to_length)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        t = self.binser.sequence_serialization( m , c )
        return t
    # end __getitem__
# end SerializedConcatDataset