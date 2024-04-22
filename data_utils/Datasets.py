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
            # check if no more melody
            if np.sum( melody[i:,:] ) == 0:
                break
            nzm = np.nonzero(m)[0]
            seq_in.append( self.melody_segment_separator )
            seq_in.extend( nzm + self.melody_offset )
        seq_in.append(self.start_harmonizing)
        for i in range(chords.shape[0]):
            # check if chord pcs exist
            c = chords[i,:]
            # check if no more chords
            if np.sum( chords[i:,:] ) == 0:
                break
            nzc = np.nonzero(c)[0]
            seq_in.append( self.chord_segment_separator )
            seq_in.extend( nzc + self.chord_offset )
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

    def indexes2labels(self, idxs):
        # input: assumes a list or numpy array of indexes between 0 and self.vocab_size
        # output: a string of labels that correspond to each index
        labels = []
        for i in idxs:
            if i == self.padding:
                labels.append( 'pad' )
            elif i == self.start_melody:
                labels.append( 'm_start' )
            elif i == self.start_harmonizing:
                labels.append( 'h_start' )
            elif i == self.melody_segment_separator:
                labels.append( 'm_seg' )
            elif i == self.chord_segment_separator:
                labels.append( 'c_seg' )
            elif i == self.end_harmonizing:
                labels.append( 'end' )
            elif i >= self.melody_offset and i < self.start_harmonizing:
                labels.append( 'm_' + str(i - self.melody_offset) )
            elif i >= self.chord_offset and i < self.end_harmonizing:
                labels.append( 'c_' + str(i - self.chord_offset) )
            else:
                labels.append( 'ERROR' )
        return labels
    # end indexes2labels

    def indexes2binary(self, idxs):
        # input: assumes a list or numpy array of indexes between 0 and self.vocab_size.
        # Additionally, input should be clearly devided in a melody and chords segment,
        # divided by a self.start_harmonizing index.
        # 
        # output:  a dict that inlcludes a message with information about errors and
        # two matrices of shape 12xM and 12xC, where M and C are the
        # length of the melody and chords respectively. If the idxs (coming from
        # a generative system) are correct, M==C, else mitigations should be followed
        # at a later stage of the pipeline.
        mel = []
        chr = []
        curr_mel = np.zeros(12)
        curr_chr = np.zeros(12)
        messages = []
        melody_started = False
        chord_started = False
        processing_melody = True
        for i in idxs:
            if i == self.padding:
                # pad should not be part of the input
                tmp_error_message = 'WARNING-pad: pad in input sequence'
                print(tmp_error_message)
                messages.append(tmp_error_message)
                # however, it may be an empty melody or chord segment...
                if processing_melody:
                    mel.append(np.zeros(12))
                else:
                    chr.append(np.zeros(12))
            elif i == self.start_melody:
                if not processing_melody:
                    tmp__error_message = 'ERROR-start_melody: currently processing chords'
                    print(tmp__error_message)
                    messages.append(tmp__error_message)
                    pass
            elif i == self.start_harmonizing:
                if not processing_melody:
                    tmp__error_message = 'ERROR-start_harmonizing: already harmonizing'
                    print(tmp__error_message)
                    messages.append(tmp__error_message)
                    pass
                else:
                    processing_melody = False
            elif i == self.melody_segment_separator:
                if not processing_melody:
                    tmp__error_message = 'ERROR-melody_segment_separator: not in processing_melody'
                    print(tmp__error_message)
                    messages.append(tmp__error_message)
                    pass
                else:
                    if melody_started:
                        mel.append( curr_mel )
                    melody_started = True
                    curr_mel = np.zeros(12)
            elif i == self.chord_segment_separator:
                if processing_melody:
                    tmp__error_message = 'ERROR-chord_segment_separator: in processing_melody'
                    print(tmp__error_message)
                    messages.append(tmp__error_message)
                    pass
                else:
                    if chord_started:
                        chr.append( curr_chr )
                    chord_started = True
                    curr_chr = np.zeros(12)
            elif i == self.end_harmonizing:
                pass
            elif i >= self.melody_offset and i < self.start_harmonizing:
                curr_mel[i - self.melody_offset] = 1
            elif i >= self.chord_offset and i < self.end_harmonizing:
                curr_chr[i - self.chord_offset] = 1
            else:
                tmp__error_message = 'ERROR-unkown label'
                print(tmp__error_message)
                messages.append(tmp__error_message)
        return {'melody': np.array(mel), 'chords': np.array(chr), 'error_messages':messages}
    # end indexes2binary
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