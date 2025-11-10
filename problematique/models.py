# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, is_bidirectional=False):
        super(trajectory2seq, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.is_bidirectional = is_bidirectional

        # Definition des couches
        # Couches pour rnn
        self.target_embedding = nn.Embedding(self.dict_size['target'], self.hidden_dim, padding_idx=symb2int['<pad>'])

        # Encoder and decoder layer can be etheir change to RNN, GRU or LSTM
        self.encoder_layer = nn.GRU(input_size=self.maxlen['input'][0], hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=self.is_bidirectional)
        self.decoder_layer = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True)

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size['target'])
        
        if self.is_bidirectional:
            self.bidirectional_merge = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.to(device)

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(x)
        
        if self.is_bidirectional:
            hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            hidden = self.bidirectional_merge(hidden)
        
        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['target'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['target'])).to(self.device) # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            vec_in = self.target_embedding(vec_in)
            dec_out, hidden = self.decoder_layer(vec_in, hidden)
            
            logits = self.fc(dec_out)  # [batch, 1, vocab_size]

            vec_out[:, i, :] = logits.squeeze(1)
            vec_in = logits.argmax(2)

        return vec_out, hidden, None

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn
    
    
class trajectory2seq_attn(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen, is_bidirectional=False):
        super(trajectory2seq_attn, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen
        self.is_bidirectional = is_bidirectional

        # Definition des couches
        # Couches pour rnn
        self.target_embedding = nn.Embedding(self.dict_size['target'], self.hidden_dim, padding_idx=symb2int['<pad>'])

        # Encoder and decoder layer can be etheir change to RNN, GRU or LSTM
        self.encoder_layer = nn.GRU(input_size=self.maxlen['input'][0], hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=self.is_bidirectional)
        self.decoder_layer = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True)


        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden2query = nn.Linear(hidden_dim, hidden_dim)

        if self.is_bidirectional:
            self.bidirectional_merge = nn.Linear(self.hidden_dim*2, self.hidden_dim)
            self.values_projection = nn.Linear(hidden_dim*2, hidden_dim)  # only if bidirectional
        
        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.dict_size['target'])

        self.to(device)
    
    def attentionModule(self, query, values):
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)
        
        if self.is_bidirectional:
            values = self.values_projection(values)  # [batch, enc_seq, hidden_dim]

        # Attention
        w = query @ values.transpose(-2, -1)
        attention_weights = nn.functional.softmax(w, dim=-1)
        attention_output = attention_weights @ values

        return attention_output, attention_weights

    def encoder(self, x):
        # Encodeur
        out, hidden = self.encoder_layer(x)

        if self.is_bidirectional:
            hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            hidden = self.bidirectional_merge(hidden)
            
        return out, hidden

    
    def decoder(self, encoder_outs, hidden):
        # Initialisation des variables
        max_len = self.maxlen['target'] # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1] # Taille de la batch
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long() # Vecteur d'entrée pour décodage 
        vec_out = torch.zeros((batch_size, max_len, self.dict_size['target'])).to(self.device) # Vecteur de sortie du décodage
        attention_weights = torch.zeros((batch_size, self.maxlen['input'][1], self.maxlen['target'])).to(self.device) # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            vec_in = self.target_embedding(vec_in)
            dec_out, hidden = self.decoder_layer(vec_in, hidden)
            
            a_a, a_w = self.attentionModule(dec_out, encoder_outs)
            attention_weights[:, :, i] = a_w.squeeze(1)
            
            # Projection dans l'espace du vocabulaire
            concat_result = torch.cat((a_a, dec_out), dim=2)
            combined = self.att_combine(concat_result)
            logits = self.fc(combined)  # [batch, 1, vocab_size]

            # Stockage de la prédiction
            vec_out[:, i, :] = logits.squeeze(1)  # correspond à la position i de la séquence

            # 5️Prédiction du prochain mot (greedy decoding)
            vec_in = logits.argmax(2)  # renvoie l'indice du mot le plus probable

            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return vec_out, hidden, attention_weights

    def forward(self, x):
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out,h)
        return out, hidden, attn

