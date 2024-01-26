import sys
sys.path.insert(0, '..')
from transformer.positional_encoding import PositionalEncoding
from torch import rand

d_model = 16

seq_length = 25
batch_size = 2

x = rand( batch_size, seq_length, d_model )

positionalEncoding = PositionalEncoding(d_model, seq_length)

y = positionalEncoding(x)

print(x.shape)
print(y.shape)