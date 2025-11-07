import numpy as np
import time
    
def edit_distance(a,b):
    # Calcul de la distance d'édition
 
    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
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

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',str(a),' et ',str(b), ': ', c)
    