from data_utils.Datasets import PermBinConcatChromaDataset
import numpy as np
from torch.utils.data import DataLoader, Subset
import sys
sys.path.insert(0, '..')
from transformer.models import ContinuousDecoderOnlyModel
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch
from tqdm import tqdm
import os
import csv

# load data
npz_path = 'data/augmented_and_padded_data.npz'
dataset = PermBinConcatChromaDataset(npz_path)

train_percentage = 0.9
split_idx = int( len(dataset)*train_percentage )

train_set = Subset(dataset, range(0,split_idx))
test_set = Subset(dataset, range(split_idx, len(dataset)))

batch_size = 8
epochs = 1000

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# permutations data
permutation_dataset = PermBinConcatChromaDataset(npz_path)
permutation_loader = DataLoader(permutation_dataset, batch_size=batch_size, shuffle=True)

# define model
vocab_size = 12
d_model = 256
num_heads = 8
num_layers = 8
d_ff = 256
seq_length = 2*129 + 1 # include "start decoding" padding - all ones
dropout = 0.3

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformer = ContinuousDecoderOnlyModel(vocab_size, d_model, num_heads, num_layers, d_ff, seq_length, dropout)

transformer = transformer.to(dev)

# train model
criterion = BCEWithLogitsLoss()
optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

save_name = 'perm_decoderOnly_binary'

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
            output = transformer(seq[:, :-1, :])
            # loss = criterion(output.contiguous().view(-1, tgt_vocab_size), chords[:, 1:, :].contiguous().view(-1, tgt_vocab_size))
            loss = criterion(output.permute(0, 2, 1), seq[:, 1:, :].permute(0, 2, 1))
            loss.backward()
            optimizer.step()
            # update loss
            samples_num += seq.shape[0]
            running_loss += loss.item()
            train_loss = running_loss/samples_num
            # accuracy
            bin_output = output > 0.5
            bin_seq = seq[:, 1:] > 0.5
            tmp_acc = 0
            tmp_count = 0
            for b_i in range(bin_output.shape[0]):
                for s_i in range(bin_output.shape[1]):
                    tmp_count += 1
                    tmp_acc += torch.all(bin_output[b_i, s_i, :].eq(bin_seq[b_i, s_i, :]))
            running_accuracy += tmp_acc.item()/tmp_count
            train_accuracy = running_accuracy/samples_num
            tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    # permutations
    perm_loss = 0
    running_loss = 0
    samples_num = 0
    running_accuracy = 0
    perm_accuracy = 0
    with tqdm(permutation_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | prm")
        for seq in tepoch:
            seq = seq.to(dev)
            optimizer.zero_grad()
            output = transformer(seq[:, :-1, :])
            loss = criterion(output.permute(0, 2, 1), seq[:, 1:, :].permute(0, 2, 1))
            loss.backward()
            optimizer.step()
            # update loss
            samples_num += seq.shape[0]
            running_loss += loss.item()
            perm_loss = running_loss/samples_num
            # accuracy
            bin_output = output > 0.5
            bin_seq = seq[:, 1:] > 0.5
            tmp_acc = 0
            tmp_count = 0
            for b_i in range(bin_output.shape[0]):
                for s_i in range(bin_output.shape[1]):
                    tmp_count += 1
                    tmp_acc += torch.all(bin_output[b_i, s_i, :].eq(bin_seq[b_i, s_i, :]))
            running_accuracy += tmp_acc.item()/tmp_count
            perm_accuracy = running_accuracy/samples_num
            tepoch.set_postfix(loss=perm_loss, accuracy=perm_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
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
            output = transformer(seq[:, :-1, :])
            # loss = criterion(output.contiguous().view(-1, tgt_vocab_size), chords[:, 1:].contiguous().view(-1, tgt_vocab_size))
            loss = criterion(output.permute(0, 2, 1), seq[:, 1:, :].permute(0, 2, 1))
            # update loss
            samples_num += seq.shape[0]
            running_loss += loss.item()
            val_loss = running_loss/samples_num
            # accuracy
            bin_output = output > 0.5
            bin_seq = seq[:, 1:] > 0.5
            tmp_acc = 0
            tmp_count = 0
            for b_i in range(bin_output.shape[0]):
                for s_i in range(bin_output.shape[1]):
                    tmp_count += 1
                    tmp_acc += torch.all(bin_output[b_i, s_i, :].eq(bin_seq[b_i, s_i, :]))
            running_accuracy += tmp_acc.item()/tmp_count
            val_accuracy = running_accuracy/samples_num
        if best_val_loss > val_loss:
            print('saving!')
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), transformer_path)
        print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
        # result_fields = ['epoch', 'train_loss', 'tran_acc', 'perm_loss', 'perm_acc', 'val_loss', 'val_acc']
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, train_accuracy, perm_loss, perm_accuracy, val_loss, val_accuracy] )