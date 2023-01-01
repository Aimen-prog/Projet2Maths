import numpy as np

"""
Projet de mathématiques:

Author: Aimen CHERIF
"""

# Partie 1 : Achats/Ventes

# Achats 1ère semaine

print("Achats de la 1ère semaine :")
#Matrice des quantités de fruits achetées par Jean
Fruits = np.array([[5, 2, 3],
                   [7, 4, 2],
                   [7, 5, 2]])
#Matrice des prix des lots de 5kg, 10kg et 15kg
Prix = np.array([[7.5, 9.5, 16.20]])
print(np.matmul(Fruits, Prix.transpose()))
TotalAchatsFruits = (np.matmul(Fruits, Prix.transpose())).sum()
print(" Total Achats 1ere semaine eneuros:", TotalAchatsFruits)

# Ventes 1ère semaine

print("Ventes de la 1ère semaine :")
#Matrice des quantités de fruits vendues par Jean durant la première semaine
Fruits = np.array([ [7.5, 7.7, 8 ],
                    [10,  11.3, 15 ],
                    [8.20, 12.4,10.5],
                    [18.9, 18,  17 ],
                    [23.2, 27.6, 22.3],
                    [26.2 , 33,  32] ])
#Matrice des prix des lots de 5kg, 10kg et 15kg
Prix = np.array([[3, 1.7, 1.9]])
print(np.matmul(Fruits, Prix.transpose()))
TotalVentes = (np.matmul(Fruits, Prix.transpose())).sum()
print("Benefices de la première semaine en euros :", TotalVentes - TotalAchatsFruits)


print("######## 2éme semaine ########")

# Achats 2ème semaine

print("Achats supplémentaires de la 2ème semaine :")

MarchandiseSupp = np.array([ [2, 2, 3, 2],
                             [3, 5, 2, 0],
                             [7, 1, 1, 2]])
Prix = np.array([[5.2, 12, 18, 21.5]])
print(np.matmul(MarchandiseSupp, Prix.transpose()))
# Total achats 2éme semaine
TotalAchats = (np.matmul(MarchandiseSupp, Prix.transpose())).sum() + TotalAchatsFruits
print("Total Achats 2e semaine en euros :", TotalAchats)

# Ventes 2ème semaine

print("Ventes de la 2ème semaine :")

A = np.array([ [5, 7, 12, 10 ],
                [6, 8, 8, 12],
                [8, 9, 11,10],
                [9, 17,16,15 ],
                [15,16,18,18],
                [17,18,20,15]])

x = np.array([[1.5], [2.5],[2.5],[3]])
b = np.array([[85, 85, 92, 141, 161.5, 165.5]]).transpose()
print(A@x)
print("les prix des kg:\n", x)


# Partie 2 : Codage/Decodage

# Codage
print("La clé")
K =  np.array([[1, 0, 0],
                [-1, 2,1],
                 [2, 1, 1]])
print("Le message crypté :")
print([12,5,19]@K, [3,12,5]@K ,[19,19,15]@K, [14,20,4]@K, [1,14,19]@K, [12,5,2]@K, [1,3,1]@K, [6,12,5]@K, [21,18,19]@K)

# Decodage
from numpy.linalg import inv
Kinv = inv(K)
print("le message décrypté :")
print([45, 29, 24]@Kinv, [1, 29 ,17]@Kinv, [30, 53, 34]@Kinv, [2 ,44, 24]@Kinv, [25, 47, 33]@Kinv, [11, 12,  7]@Kinv, [0, 7 ,4]@Kinv, [4, 29, 17]@Kinv, [41, 55, 37]@Kinv)

# Partie 3 : Retardite (Diagonalisation)
R =    np.array([[0.7, 0.3],
                   [0.2, 0.8]])
print("valeurs propres :")
w = np.linalg.eigvals(R)
print(w)
v,w= np.linalg.eig(R)

print("10e génération")
D=np.array([[0.5, 0],
            [0, 1]])
print(w @ np.linalg.matrix_power(D, 10) @  inv(w))

print("Vérification :")
print(np.linalg.matrix_power(R, 10))
