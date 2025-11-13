# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *
from dataset import HandwrittenWords 
from models import trajectory2seq, trajectory2seq_attn
from metrics import edit_distance, confusion_matrix

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = False           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    display_attention = True
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    batch_size = 32
    train_val_split = .7        # Ratio des echantillions pour l'entrainement
    n_hidden = 20               # Nombre de neurones caches par couche 
    n_layers = 2               # Nombre de de couches

    # À compléter
    n_epochs = 50
    lr = 0.01                   # Taux d'apprentissage pour l'optimizateur
    model_name = 'model_gru_attn_bid'

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # À compléter
    dataset = HandwrittenWords('problematique/data_trainval.p')

    
    # Séparation de l'ensemble de données (entraînement et validation)
    # À compléter
    n_train_samp = int(len(dataset)*train_val_split)
    n_val_samp = len(dataset)-n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])
   

    # Instanciation des dataloaders
    # À compléter
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    # Instanciation du model
    # À compléter

    best_val_loss = np.inf # pour sauvegarder le meilleur model
    if trainning:
        model = trajectory2seq(hidden_dim=n_hidden, \
        n_layers=n_layers, device=device, symb2int=dataset.symb2int, \
        int2symb=dataset.int2symb, dict_size=dataset.dict_size, maxlen=dataset.max_len, is_bidirectional=False)

        # Afficher le résumé du model
        print('Model : \n', model, '\n')
        print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

        # Fonction de coût et optimizateur
        # À compléter
        if learning_curves:
            train_dist =[] # Historique des distances de train
            train_loss=[] # Historique des coûts de train
            val_dist =[] # Historique des distances de val
            val_loss=[] # Historique des coûts da val
            fig, ax = plt.subplots(2) # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symb2int[dataset.pad_symbol]) # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            model.train()
            running_loss_train = 0
            dist=0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad() # Mise a zero du gradient
                output, hidden, attn = model(input_seq)# Passage avant
                loss = criterion(output.view((-1, model.dict_size['target'])), target_seq.view(-1))
                
                loss.backward() # calcul du gradient
                optimizer.step() # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output,dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = len(a)
                    Mb = len(b)
                    dist += edit_distance(a[:Ma],b[:Mb])/batch_size


                # Affichage pendant l'entraînement
                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')
            print('\n')
            
            running_loss_val = 0
            running_val_dist = 0
            model.eval()
            for data in dataload_val:
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()
                
                output, hidden, attn = model(input_seq)# Passage avant
                loss = criterion(output.view((-1, model.dict_size['target'])), target_seq.view(-1))
                
                running_loss_val += loss.item()
                
                output_list = torch.argmax(output,dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = len(a)
                    Mb = len(b)
                    running_val_dist += edit_distance(a[:Ma],b[:Mb])/batch_size


            print('\nValidation - Average loss: {:.4f}'.format(running_loss_val/len(dataload_val)))
            print('')

            # Affichage
            if learning_curves:
                # training metrics
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist/len(dataload_train))
                
                # validation metrics
                val_loss.append(running_loss_val/len(dataload_val))
                val_dist.append(running_val_dist/len(dataload_val))
                
                ax[0].cla()
                ax[1].cla()
                ax[0].plot(train_loss, label='training loss')
                ax[1].plot(train_dist, label='training distance')
                ax[0].plot(val_loss, label='validation loss')
                ax[1].plot(val_dist, label='validation distance')
                ax[0].legend()
                ax[1].legend()
                plt.draw()
                plt.pause(0.001)
                plt.savefig(f'problematique/figures/{model_name}_training.png')
            
            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model,f'problematique/{model_name}.pt')
            
        if learning_curves:
            plt.show()
            plt.close('all')

    if test:
        # Évaluation
        # À compléter
        model = torch.load(f'problematique/{model_name}.pt', weights_only=False)
        model = model.to(device).float()
        model.eval()

        # Afficher le résumé du model
        print('Model : \n', model, '\n')
        print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

        # Charger les données de tests
        # À compléter
        dataset_test = HandwrittenWords('problematique/data_test.p', max_len=model.maxlen)
        dataload_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        
        n_classes = dataset_test.dict_size['target']
        conf_mat = np.zeros((n_classes, n_classes), dtype=int)

        total_edit_dist = 0
        total_inference_time = 0

        # Boucle de test
        with torch.no_grad():
            for data in dataload_test:
                input_seq, target_seq = data
                input_seq = input_seq.to(device).float()
                target_seq = target_seq.to(device).long()

                start_time = time.time()
                output, hidden, attn = model(input_seq)
                end_time = time.time()
                total_inference_time += end_time - start_time

                preds = torch.argmax(output, dim=-1).cpu().numpy()
                targets = target_seq.cpu().numpy()

                # Mise à jour de la matrice de confusion (lettre par lettre)
                conf_mat = confusion_matrix(targets, preds, conf_mat, ignore=[dataset_test.symb2int[dataset_test.pad_symbol], dataset_test.symb2int[dataset_test.start_symbol], dataset_test.symb2int[dataset_test.stop_symbol]])

                # Calcul de la distance d'édition moyenne
                M = len(preds)
                for i in range(M):
                    a = targets[i]
                    b = preds[i]
                    Ma = len(a)
                    Mb = len(b)
                    total_edit_dist += edit_distance(a[:Ma],b[:Mb])/batch_size

        avg_edit_dist = total_edit_dist / len(dataload_test)
        print(f"Distance d'édition moyenne sur le test : {avg_edit_dist:.3f}")

        avg_inference_time = total_inference_time / len(dataload_test)
        print(f"Temps d'inférence moyenne sur le test : {avg_inference_time:.3f}")

        # Normaliser la matrice de confusion pour affichage
        conf_mat_norm = conf_mat / (conf_mat.sum(axis=1, keepdims=True) + 1e-8)
        conf_mat_norm = np.nan_to_num(conf_mat_norm)  # éviter les NaN
        
        for num in range(10):
            idx = np.random.randint(0,len(dataset_test))
            input_seq, target_seq = dataset_test[idx]
            
            input_seq = input_seq.to(device).float()
            target_seq = target_seq.to(device).long()
            
            output, hidden, attn = model(input_seq.view(1,-1, 2))
            out = torch.argmax(output,dim=-1).detach().cpu().tolist()
            
            out_seq = [model.int2symb[i] for i in out[0]]
            target = [model.int2symb[i] for i in target_seq.detach().cpu().tolist()]
            # Affichage de l'attention
            # À compléter (si nécessaire)

            # Affichage des résultats de test
            # À compléter
            print('\nTarget: ', ' '.join(target))
            print('Output: ', ' '.join(out_seq))
            
            # Affichage de la matrice de confusion
            # À compléter
            
        import matplotlib.pyplot as plt

        labels = [model.int2symb[i] for i in range(1, 27)]
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(conf_mat_norm, cmap="Blues")
        plt.title("Matrice de confusion normalisée (Test)", pad=20)
        fig.colorbar(cax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité terrain")

        # Afficher les valeurs numériques dans la matrice
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = conf_mat_norm[i, j]
                if val > 0.001:  # éviter d'afficher les zéros
                    ax.text(j, i, f"{val:.2f}", va='center', ha='center', color='black', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'problematique/figures/{model_name}_conf.png')
        plt.show()
        