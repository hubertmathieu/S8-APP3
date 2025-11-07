# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super(Model, self).__init__()
        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        self.batch_first = True
        self.rnn = nn.RNN(input_size=1, hidden_size=n_hidden, num_layers=n_layers, batch_first=self.batch_first)
        self.fc = nn.Linear(in_features=n_hidden, out_features=1)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
    
    def forward(self, x, hx=None):
        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------        
        output, hx = self.rnn(x, hx)
        output = self.fc(output)
        output = torch.tanh(output)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------

        return output, hx

if __name__ == '__main__':
    x = torch.zeros((100,2,1)).float()
    model = Model(25)
    print(model(x))