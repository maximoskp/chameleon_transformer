from .BinaryTokenizer import BinaryTokenizer
from torch.utils.data import Dataset
import numpy as np

class BinarySerializer:
    def __init__(self, pad_to_length=0, left_padding=True):
        '''
        0: padding
        1: start_melody
        2: melody segment seperator
        3 to 3+11=14: melody pitch class
        15: start harmonizing
        16: chord segment separator
        17 to 17+11=28: chord pitch class
        29: end harmonizing
        
        vocab_size = 30
        '''
        self.padding = 0
        self.start_melody = 1
        self.melody_segment_separator = 2
        self.melody_offset = 3
        self.start_harmonizing = 15
        self.chord_segment_separator = 16
        self.chord_offset = 17
        self.end_harmonizing = 29
        self.max_seq_length = 0
        self.pad_to_length = pad_to_length
        self.left_padding = left_padding
        self.vocab_size = 30
    # end init
    def sequence_serialization(self, melody, chords):
        seq_in = []
        seq_in.append(self.start_melody)
        for i in range(melody.shape[0]):
            # check if melody pcs exist
            m = melody[i,:]
            nzm = np.nonzero(m)[0]
            seq_in.append( self.melody_segment_separator )
            seq_in.extend( nzm + self.melody_offset )
            # check if no more melody
            if np.sum( melody[i:,:] ) == 0:
                break
        seq_in.append(self.start_harmonizing)
        for i in range(chords.shape[0]):
            # check if chord pcs exist
            c = chords[i,:]
            nzc = np.nonzero(c)[0]
            seq_in.append( self.chord_segment_separator )
            seq_in.extend( nzc + self.chord_offset )
            # check if no more chords
            if np.sum( chords[i:,:] ) == 0:
                break
        seq_in.append(self.end_harmonizing)
        if len(seq_in) > self.max_seq_length:
            self.max_seq_length = len(seq_in)
        seq_in_np = np.array(seq_in)
        if seq_in_np.shape[0] < self.pad_to_length:
            # left padding
            if self.left_padding:
                seq_in_np = np.pad(seq_in_np, (self.pad_to_length - seq_in_np.shape[0], 0), constant_values=(self.padding, self.padding))
            else:
                seq_in_np = np.pad(seq_in_np, (0, self.pad_to_length - seq_in_np.shape[0]), constant_values=(self.padding, self.padding))
        # masks
        target_masked = np.array(seq_in_np[1:])
        target_masked[target_masked < self.start_harmonizing] = -100
        return seq_in_np, target_masked
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
    def __init__(self, npz_path, pad_to_length=1100, left_padding=True):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.binser = BinarySerializer(pad_to_length=pad_to_length, left_padding=left_padding)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        for i in range(m.shape[0]):
            p = np.random.permutation(12)
            m[i,:] = m[i,p]
            c[i,:] = c[i,p]
        t, t_masked = self.binser.sequence_serialization( m , c )
        return t, t_masked
    # end __getitem__
# end PermSerializedConcatDataset

class ShiftSerializedConcatDataset(Dataset):
    def __init__(self, npz_path, pad_to_length=1100, left_padding=True):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.binser = BinarySerializer(pad_to_length=pad_to_length, left_padding=left_padding)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        # get length of c reducing its size
        len_c = c.shape[0]
        # random end, keep at least 1
        idx = np.random.randint(len_c-1) + 1
        t, t_masked = self.binser.sequence_serialization( m , c[:idx,:] )
        return t, t_masked
    # end __getitem__
# end ShiftSerializedConcatDataset

class MelBoostSerializedConcatDataset(Dataset):
    def __init__(self, npz_path, pad_to_length=1100, left_padding=True, remove_percentage=0.3):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.remove_percentage = remove_percentage
        self.binser = BinarySerializer(pad_to_length=pad_to_length, left_padding=left_padding)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        # remove random components of the melody
        # find non-zero indexes
        nnz = np.nonzero(m)
        # get number of non zero
        num_nnz = nnz[0].size
        # remove percentage
        num_remove = int(num_nnz*self.remove_percentage)
        # permutate idxs
        perm_idxs = np.random.permutation(num_nnz)
        # zero-out first permutated
        m[ nnz[0][perm_idxs[:num_remove]] , nnz[1][perm_idxs[:num_remove]] ] = 0
        t, t_masked = self.binser.sequence_serialization( m , c )
        return t, t_masked
    # end __getitem__
# end MelBoostSerializedConcatDataset

class SerializedConcatDataset(Dataset):
    def __init__(self, npz_path, pad_to_length=1100, left_padding=True):
        data = np.load(npz_path)
        self.melody_pcps = data['melody_pcps'].astype('float32')
        self.chord_pcps = data['chord_pcps'].astype('float32')
        self.binser = BinarySerializer(pad_to_length=pad_to_length, left_padding=left_padding)
    # end __init__
    
    def __len__(self):
        return self.melody_pcps.shape[0]
    # end __len__
    
    def __getitem__(self, idx):
        m = self.melody_pcps[idx,:,:]
        c = self.chord_pcps[idx,:,:]
        t, t_masked = self.binser.sequence_serialization( m , c )
        return t, t_masked
    # end __getitem__
# end SerializedConcatDataset