import sys
sys.path.insert(0, '..')
from transformer.layers import DecoderLayer
from torch import rand

d_model = 16
num_heads = 4
d_ff = 8
dropout = 0.3

seq_length = 25
batch_size = 2

x = rand( batch_size, seq_length, d_model )
enc_output = rand( batch_size, seq_length, d_model )

encoder = DecoderLayer(d_model, num_heads, d_ff, dropout)

y = encoder(x, enc_output)

print(x.shape)
print(y.shape)