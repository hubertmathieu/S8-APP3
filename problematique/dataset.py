import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename, max_len = None):
        # Lecture du text
        self.sequence_max_len = 5
        self.pad_symbol = '<pad>'
        self.start_symbol = '<sos>'
        self.stop_symbol = '<eos>'
        self.symb2int = dict()
        self.int2symb = dict()
        self.dict_size = dict()
        self.max_len = dict()
        
        self.input_padding = -1
        self.input_eos = -2

        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        # Extraction des symboles
        # À compléter
        # data = ['lynn' = [[x...],[y...]], ...]
        # data = [5, 7, 12, ... ] = [[x...],[y...]]
        self.symb2int[self.start_symbol] = 0
        for i in range(1, 27):
            # {'a' = 0, 'b', ..}
            self.symb2int[f'{chr(i + 96)}'] = i
        
        self.symb2int[self.pad_symbol] = 27
        self.symb2int[self.stop_symbol] = 28

        self.int2symb = {v:k for k,v in self.symb2int.items()}

        self.temp_data = {}
        self.temp_data['input'] = {}
        self.temp_data['target'] = {}
        
        max_len_padding_target = self.sequence_max_len + 1
        if max_len is None:
            self.max_len['input'] = (len(self.data[0][1]), max(len(self.data[i][1][0]) for i in range(len(self.data))))
            self.max_len['target'] =  max_len_padding_target
        else:
            self.max_len = max_len
        
        # padding
        for i in range(len(self.data)):
            word, coordinates = self.data[i]
            
            # padding on target (letters)
            target_sequence = np.zeros(max_len_padding_target)
            for y in range(max_len_padding_target):
                if y < len(word):
                    letter = word[y]
                    target_sequence[y] = self.symb2int[letter]
                elif y == len(word):
                    target_sequence[y] = self.symb2int[self.stop_symbol]
                else:
                    target_sequence[y] = self.symb2int[self.pad_symbol]
            
            # padding on input (coordinates)
            coord_length = len(coordinates[0])
            input_seq = np.zeros(self.max_len['input'], dtype=float)
            
            for j in range(self.max_len['input'][1]):
                if j >= coord_length: # padding
                    input_seq[0][j] = input_seq[0][j-1]
                    input_seq[1][j] = input_seq[1][j-1]
                else: # values
                    input_seq[0][j] = coordinates[0][j]
                    input_seq[1][j] = coordinates[1][j]

            # Example: normalize each sequence between -1 and 1
            input_seq = input_seq.T
            input_seq = self.normalize_coords(input_seq)
            
            self.temp_data['input'][i] = input_seq
            self.temp_data['target'][i] = target_sequence

        
        self.data = self.temp_data
        self.dict_size = {'target':len(self.symb2int)}
        
    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, idx):
        # À compléter
        input_seq = self.data['input'][idx]
        target_seq = self.data['target'][idx]
       
        return input_seq.clone().detach(), torch.tensor(target_seq) # (2, 457), (6)
    
    def normalize_coords(self, input_seq):
        # input_seq: shape (T, 2)
        x = input_seq[:, 0]
        y = input_seq[:, 1]

        # Center and scale each coordinate channel
        x = torch.tensor((x - x.mean()) / (x.std() + 1e-8))
        y = torch.tensor((y - y.mean()) / (y.std() + 1e-8))

        input_seq = torch.stack((x, y), dim=1)  # shape (T, 2)
        
        return input_seq

    def visualisation(self, idx):
        # Visualisation des échantillons
        # À compléter (optionel)
        # Retrieve the input coordinates and target sequence
        input_seq = self.data['input'][idx].T      # [[x...], [y...]]
        target_seq = self.data['target'][idx]    # encoded sequence
        
        # Extract x and y
        x, y = input_seq

        # Convert numeric target sequence back to symbols for display
        int2symb = {v: k for k, v in self.symb2int.items()}
        decoded_word = ''.join(
            int2symb[int(i)] for i in target_seq 
            if int2symb[int(i)] not in ['<pad>', '<sos>', '<eos>']
        )

        # Plot the handwritten word trajectory
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, marker='o', color='blue', linewidth=1)
        plt.title(f"Index {idx} → '{decoded_word}'")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
        pass
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('problematique/data_trainval.p')
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))