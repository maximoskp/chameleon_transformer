from data_utils.Datasets import SerializedConcatDataset
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import sys
sys.path.insert(0, '..')
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from tqdm import tqdm
import os
import csv
import pickle

from transformers import AutoConfig, GPT2LMHeadModel

with open('tests/serializer_jazz.pkl', 'rb') as inp:
    binser = pickle.load(inp)

# define model
vocab_size = binser.vocab_size
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 256
max_seq_length = binser.max_seq_length
dropout = 0.3

# load data
# npz_path = 'data/augmented_and_padded_data.npz'
npz_path = 'data/augmented_and_padded_data_650_songs_with_measure_info.npz'
dataset = SerializedConcatDataset(npz_path, pad_to_length=max_seq_length, left_padding=False)

train_percentage = 0.9
split_idx = int( len(dataset)*train_percentage )

# shuffle before split
generator = torch.Generator().manual_seed(42)

train_set, test_set = random_split(dataset, [split_idx, len(dataset)-split_idx], generator=generator)

# train_set = Subset(dataset, range(0,split_idx))
# test_set = Subset(dataset, range(split_idx, len(dataset)))

batch_size = 4
epochs = 1000

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# no permutation just another copy of the dataset
shift_dataset = SerializedConcatDataset(npz_path, pad_to_length=max_seq_length, left_padding=False)
shift_loader = DataLoader(shift_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=vocab_size,
    n_positions=max_seq_length,
    n_layer=num_layers,
    n_head=num_heads,
    pad_token_id=binser.padding,
    bos_token_id=binser.padding,
    eos_token_id=binser.padding,
    n_embd=d_ff
)
transformer = GPT2LMHeadModel(config).to(dev)

transformer = transformer.to(dev)

# train model
criterion = CrossEntropyLoss(ignore_index=-100)
optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

save_name = 'jazz_GPT2'

# keep best validation loss for saving
best_val_loss = np.inf
save_dir = 'saved_models/' + save_name + '/'
transformer_path = save_dir + save_name + '.pt'
os.makedirs(save_dir, exist_ok=True)

# save results
os.makedirs('results', exist_ok=True)
results_path = 'results/' + save_name + '.csv'
result_fields = ['epoch', 'train_loss', 'tran_acc', 'shift_loss', 'shift_acc', 'val_loss', 'val_acc']
with open( results_path, 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow( result_fields )

for epoch in range(epochs):
    train_loss = 0
    running_loss = 0
    batch_num = 0
    running_accuracy = 0
    train_accuracy = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | trn")
        for seq, masked_target in tepoch:
            seq = seq.to(dev)
            masked_target = masked_target.to(dev)
            optimizer.zero_grad()
            output = transformer(seq[:, :-1], attention_mask=seq[:,:-1] != 0)
            loss = criterion(output.logits.contiguous().view(-1, vocab_size), masked_target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            # update loss
            batch_num += 1
            running_loss += loss.item()
            train_loss = running_loss/batch_num
            # accuracy
            prediction = output.logits.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction[masked_target >= binser.start_harmonizing] == masked_target[masked_target >= binser.start_harmonizing]).sum().item()/(masked_target >= binser.start_harmonizing).sum().item()
            train_accuracy = running_accuracy/batch_num
            torch.set_printoptions(threshold=10_000)
            tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    # shifts
    shift_loss = 0
    running_loss = 0
    batch_num = 0
    running_accuracy = 0
    shift_accuracy = 0
    with tqdm(shift_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | prm")
        for seq, masked_target in tepoch:
            seq = seq.to(dev)
            output = transformer(seq[:, :-1], attention_mask=seq[:,:-1] != 0)
            masked_target = masked_target.to(dev)
            optimizer.zero_grad()
            loss = criterion(output.logits.contiguous().view(-1, vocab_size), masked_target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            # update loss
            batch_num += 1
            running_loss += loss.item()
            shift_loss = running_loss/batch_num
            # accuracy
            prediction = output.logits.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction[masked_target >= binser.start_harmonizing] == masked_target[masked_target >= binser.start_harmonizing]).sum().item()/(masked_target >= binser.start_harmonizing).sum().item()
            shift_accuracy = running_accuracy/batch_num
            tepoch.set_postfix(loss=shift_loss, accuracy=shift_accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    # validation
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        print('validation...')
        for seq, masked_target in tepoch:
            seq = seq.to(dev)
            output = transformer(seq[:, :-1], attention_mask=seq[:,:-1] != 0)
            masked_target = masked_target.to(dev)
            optimizer.zero_grad()
            loss = criterion(output.logits.contiguous().view(-1, vocab_size), masked_target.contiguous().view(-1))
            # update loss
            batch_num += 1
            running_loss += loss.item()
            val_loss = running_loss/batch_num
            # accuracy
            prediction = output.logits.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction[masked_target >= binser.start_harmonizing] == masked_target[masked_target >= binser.start_harmonizing]).sum().item()/(masked_target >= binser.start_harmonizing).sum().item()
            val_accuracy = running_accuracy/batch_num
        if best_val_loss > val_loss:
            print('saving!')
            best_val_loss = val_loss
            torch.save(transformer.state_dict(), transformer_path)
        print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
        # result_fields = ['epoch', 'train_loss', 'tran_acc', 'shift_loss', 'shift_acc', 'val_loss', 'val_acc']
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, train_accuracy, shift_loss, shift_accuracy, val_loss, val_accuracy] )
