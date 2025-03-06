"""
POLYTECHNIQUE MONTRÉAL
MEC8211 : VÉRIFICATION ET VALIDATION EN MODÉLISATION NUMÉRIQUE

DEVOIR 2 : VÉRIFICATION DE CODE PAR LA MÉTHODE DES SOLUTIONS MANUFACTURÉES

PAR : Hugo Trépanier, Jean Zhu & Renno Soussa
"""

"""
Ce code alimente les données d'entrée au code de résolution de l'équation de diffusion-réaction du Devoir 2.
"""

# Données d'entrée :

k = 4*10**(-9) # s^-1 (Constante de réaction d'ordre 1)
Deff = 10**(-10) # m^2/s (Coefficient de diffusion effectif du sel dans le béton)
Ce = 20 # mol/m^3 (Concentration du sel à la surface du béton)
N = 11 # noeuds (Nombre de noeuds dans la dimension spatiale du problème)
Ntemps = 10 # pas de temps (Nombre de pas de temps du problème transitoire)