from data_utils.Datasets import TokenizedConcatChromaDataset
import numpy as np
from torch.utils.data import DataLoader, Subset
import sys
sys.path.insert(0, '..')
from transformer.models import DecoderOnlyModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from tqdm import tqdm
import os
import csv

npz_path = 'data/augmented_and_padded_data.npz'
dataset = TokenizedConcatChromaDataset(npz_path)

train_percentage = 0.9
split_idx = int( len(dataset)*train_percentage )

train_set = Subset(dataset, range(0,split_idx))
test_set = Subset(dataset, range(split_idx, len(dataset)))

batch_size = 16
epochs = 1000

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

vocab_size = 2**12
d_model = 256
num_heads = 8
num_layers = 8
d_ff = 256
max_seq_length = 2*129 + 1 # include "start decoding" padding - all ones
dropout = 0.3

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformer = DecoderOnlyModel(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

transformer = transformer.to(dev)

criterion = CrossEntropyLoss(ignore_index=0)
optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

save_name = 'decoderOnly_one_hot'

# keep best validation loss for saving
best_val_loss = np.inf
save_dir = 'saved_models/' + save_name + '/'
transformer_path = save_dir + save_name + '.pt'
os.makedirs(save_dir, exist_ok=True)

# save results
os.makedirs('results', exist_ok=True)
results_path = 'results/' + save_name + '.csv'
result_fields = ['epoch', 'train_loss', 'tran_acc', 'perm_loss', 'perm_acc', 'val_loss', 'val_acc']
with open( results_path, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow( result_fields )

for epoch in range(epochs):
    train_loss = 0
    running_loss = 0
    samples_num = 0
    running_accuracy = 0
    train_accuracy = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | trn")
        for seq in tepoch:
            seq = seq.to(dev)
            optimizer.zero_grad()
            output = transformer(seq[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), seq[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            # update loss
            samples_num += seq.shape[0]
            running_loss += loss.item()
            train_loss = running_loss/samples_num
            # accuracy
            prediction = output.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction == seq[:, 1:]).sum().item()/prediction.shape[1]
            train_accuracy = running_accuracy/samples_num
            tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    # validation
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        samples_num = 0
        running_accuracy = 0
        val_accuracy = 0
        print('validation...')
        for seq in test_loader:
            seq = seq.to(dev)
            output = transformer(seq[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), seq[:, 1:].contiguous().view(-1))
            # update loss
            samples_num += seq.shape[0]
            running_loss += loss.item()
            val_loss = running_loss/samples_num
            # accuracy
            prediction = output.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction == seq[:, 1:]).sum().item()/prediction.shape[1]
            val_accuracy = running_accuracy/samples_num
        if best_val_loss > val_loss:
            print('saving!')
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), transformer_path)
        print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
        # result_fields = ['epoch', 'train_loss', 'tran_acc', 'val_loss', 'val_acc']
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, train_accuracy, val_loss, val_accuracy] )