"""
POLYTECHNIQUE MONTRÉAL
MEC8211 : VÉRIFICATION ET VALIDATION EN MODÉLISATION NUMÉRIQUE

DEVOIR 2 : VÉRIFICATION DE CODE PAR LA MÉTHODE DES SOLUTIONS MANUFACTURÉES

PAR : Hugo Trépanier, Jean Zhu & Renno Soussa
"""

"""
Ce code permet de résoudre l'équation de diffusion-réaction transitoire impliquée dans le problème de diffusion du sel minéral dissout dans l'eau
à travers un pilier de béton en contact avec l'eau saline.

L'objectif est d'obtenir le profil de concentration en coordonnées cylindrique selon la position dans le rayon du pilier de béton, et ce, pour 
plusieurs pas de temps. Cela permet alors de caractériser l'évolution temporelle du transfert de masse dans le pilier.

La résolution spatiale se fait par la méthode des différences finies avec des schémas de différentiation d'ordre 2 pour permettre une résolution exacte
de l'équation différentielle spatiale, et ce, puisque la solution analytique est un polynôme d'ordre 2.
La résolution temporelle se fait par la méthode de XXXXXXXXXXXXXXXXXX puisque XXXXXXXXXXXXXXXXXXX

À la fin de la résoltion, nous nous attendons à obtenir un graphique du profil de concentration selon la position dans le rayon pour à plusieurs temps t.
Le profil parabolique sera plus prononcé aux premiers pas de temps et et s'aplatira au fil du temps. Une réaction d'ordre 1 consommant le sel dans le pilier
existe, donc, le profil parabolique ne deviendra probablement pas complètement plat puisqu'un gradient de concentration se maintiendra.

"""

# Importation des librairies pertinentes
import numpy as np
import matplotlib.pyplot as plt

"""
Fontions utilisées dans la résolution de l'équation différentielle
"""
# Création de la matrice de discrétisation spatiale de l'e.d.p.
def make_A_matrix_order2(N, R):
    "Génération de la matrice A"
    dr = R/(N-1)
    A = np.zeros((N, N))
    A[0,0:3] = [-3/(2*dr), 4/(2*dr), -1/(2*dr)]

    for i in range(1, N-1):
        r = dr*i
        A[i, i-1:i+2] = [(-1/(2*r*dr) + 1/(dr**2)), -2/(dr**2), 1/(2*r*dr) + 1/(dr**2)]

    A[N-1, N-1] = 1

    return A




# Objets contenant les noeuds et les erreurs L1, L2 et Linfini
vect_N= []
L1_1 = []
L1_2 = []
L2_1 = []
L2_2 = []
Linf_1 = []
Linf_2 = []