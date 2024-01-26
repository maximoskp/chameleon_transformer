import sys
sys.path.insert(0, '..')
from transformer.attention import MultiHeadAttention
from torch import rand

d_model = 16
num_heads = 4

seq_length = 25
batch_size = 2

queries = rand( batch_size, seq_length, d_model )
keys = rand( batch_size, seq_length, d_model )
values = rand( batch_size, seq_length, d_model )

multihead = MultiHeadAttention(d_model, num_heads)

x = multihead(queries, keys, values)

print(x.shape)