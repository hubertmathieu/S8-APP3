# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(a,b):
    # Calcul de la distance d'édition
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
       
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
 
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            condition_1 = dp[i-1][j]+1
            condition_2 = dp[i][j-1]+1
            condition_3 = dp[i-1][j-1]
 
            if a[i-1] != b[j-1]:
                condition_3 += 1
           
            dp[i][j] = min(condition_1, condition_2, condition_3)
 
    return dp[len_a][len_b]

def confusion_matrix(true, pred, conf_mat, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter
    for t, p in zip(true, pred):
        for true_char, pred_char in zip(t, p):
            if true_char not in ignore:  # ignorer les <pad>
                conf_mat[true_char, pred_char] += 1

    return conf_mat
