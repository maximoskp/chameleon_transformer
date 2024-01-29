from data_utils.Datasets import TokenizedChromaDataset
import numpy as np
from torch.utils.data import DataLoader, Subset
import sys
sys.path.insert(0, '..')
from transformer.models import TransformerFromModels, EncoderModel, DecoderModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from tqdm import tqdm
import os

npz_path = 'data/augmented_and_padded_data.npz'
dataset = TokenizedChromaDataset(npz_path)

train_percentage = 0.9
split_idx = int( len(dataset)*train_percentage )

train_set = Subset(dataset, range(0,split_idx))
test_set = Subset(dataset, range(split_idx, len(dataset)))

batch_size = 16
epochs = 1000

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

src_vocab_size = 2**12
tgt_vocab_size = 2**12
d_model = 128
num_heads = 4
num_layers = 4
d_ff = 128
max_seq_length = 129
dropout = 0.3

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoderModel = EncoderModel(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
decoderModel = DecoderModel(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

encoderModel = encoderModel.to(dev)
decoderModel = decoderModel.to(dev)

transformer = TransformerFromModels(encoderModel, decoderModel)

transformer = transformer.to(dev)

criterion = CrossEntropyLoss(ignore_index=0)
optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# keep best validation loss for saving
best_val_loss = np.inf
save_dir = 'saved_models/encoder_decoder_one_hot/'
encoder_path = save_dir + 'encoder_one_hot.pt'
decoder_path = save_dir + 'decoder_one_hot.pt'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    train_loss = 0
    running_loss = 0
    samples_num = 0
    running_accuracy = 0
    accuracy = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch} | trn")
        for melodies, chords in tepoch:
            melodies = melodies.to(dev)
            chords = chords.to(dev)
            optimizer.zero_grad()
            output = transformer(melodies, chords[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), chords[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            # update loss
            samples_num += melodies.shape[0]
            running_loss += loss.item()
            train_loss = running_loss/samples_num
            # accuracy
            prediction = output.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction == chords[:, 1:]).sum().item()/prediction.shape[1]
            accuracy = running_accuracy/samples_num
            tepoch.set_postfix(loss=train_loss, accuracy=accuracy) # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
    # validation
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        samples_num = 0
        running_accuracy = 0
        accuracy = 0
        print('validation...')
        for melodies, chords in test_loader:
            melodies = melodies.to(dev)
            chords = chords.to(dev)
            output = transformer(melodies, chords[:, :-1])
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), chords[:, 1:].contiguous().view(-1))
            # update loss
            samples_num += melodies.shape[0]
            running_loss += loss.item()
            val_loss = running_loss/samples_num
            # accuracy
            prediction = output.argmax(dim=2, keepdim=True).squeeze()
            running_accuracy += (prediction == chords[:, 1:]).sum().item()/prediction.shape[1]
            accuracy = running_accuracy/samples_num
        if best_val_loss > val_loss:
            print('saving!')
            best_val_loss = val_loss
            torch.save(encoderModel.state_dict(), encoder_path)
            torch.save(decoderModel.state_dict(), decoder_path)
        print(f'validation: accuracy={accuracy}, loss={val_loss}')