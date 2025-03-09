# Fichier: analyse_de_convergence.py
#
# But: Tracer un graphique d'analyse de convergence avec regression en loi de puissance à partir des 
#      données contenues dans les fichiers "erreurs" et liste_des_resolutions. Ce dernier nom et les variables
#      0 et 100.0 seront remplacés par le script bash analyse_auto qui appelle ce programme python.

import numpy as np
import matplotlib.pyplot as plt

# Fonction pour lire des fichiers de données
def reading_files_N(erreurs):

    # Lecture des erreurs absolues
    with open(erreurs, 'r') as fichier1:
        error_values = [float(ligne) for ligne in fichier1.read().splitlines()]

    # Lecture de nombre d'intervalle utilisée pour l'intégration dans le fichier passé en 2eme argument du script bash
    with open('NOM_DU_FICHIER', 'r') as fichier2:
        nombres = [int(ligne) for ligne in fichier2.read().splitlines()]

    # Calcul du pas d'intégration (Δx = h) en fonction des bornes qui se trouvemt dans le fichier trapezoidal.cpp
    h_values = [(RAYON_B - RAYON_A) / nombre for nombre in nombres] 

    # Trier `h_values` par ordre croissant et appliquer le même ordre à `error_values`
    paires_triees = sorted(zip(h_values, error_values))

    # Déballer les paires triées
    h_values_triees, error_values_triees = zip(*paires_triees)

    # Convertir les tuples en listes
    h_values = list(h_values_triees)
    error_values = list(error_values_triees)

    return h_values, error_values
    
def reading_files_T(erreurs):

    # Lecture des erreurs absolues
    with open(erreurs, 'r') as fichier1:
        error_values = [float(ligne) for ligne in fichier1.read().splitlines()]

    # Lecture de nombre d'intervalle utilisée pour l'intégration dans le fichier passé en 2eme argument du script bash
    with open('NOM_DU_FICHIER', 'r') as fichier2:
        nombres = [int(ligne) for ligne in fichier2.read().splitlines()]

    # Calcul du pas d'intégration (Δx = h) en fonction des bornes qui se trouvemt dans le fichier trapezoidal.cpp
    t_values = [(TEMPS_B - TEMPS_A) / nombre for nombre in nombres] 

    # Trier `h_values` par ordre croissant et appliquer le même ordre à `error_values`
    paires_triees = sorted(zip(t_values, error_values))

    # Déballer les paires triées
    t_values_triees, error_values_triees = zip(*paires_triees)

    # Convertir les tuples en listes
    t_values = list(t_values_triees)
    error_values = list(error_values_triees)

    return t_values, error_values

def graph_convergence(maillages, erreurs, type, dimension, unite):
    coefficients = np.polyfit(np.log(maillages), np.log(erreurs), 1)
    exponent = coefficients[0]

    # Fonction de rÃ©gression en termes de logarithmes
    fit_function_log = lambda x: exponent * x + coefficients[1]

    # Fonction de rÃ©gression en termes originaux
    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

    # Extrapoler la valeur prÃ©dite pour la derniÃ¨re valeur de h_values
    extrapolated_value = fit_function(maillages[-1])

    # Tracer le graphique en Ã©chelle log-log avec des points et la courbe de rÃ©gression extrapolÃ©e
    plt.figure(figsize=(8, 6))
    plt.scatter(maillages, erreurs, marker='o', color='b', label='DonnÃ©es numÃ©riques obtenues')
    plt.loglog(maillages, fit_function(maillages), linestyle='--', color='r', label='RÃ©gression en loi de puissance')
    
    # Ajouter des Ã©tiquettes et un titre au graphique
    plt.title('Convergence d\'ordre 1\n de l\'erreur 'f'{type}'  ' en fonction de ' f'{dimension}',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramÃ¨tre y rÃ¨gle la position verticale du titre

    plt.xlabel('Pas de diffÃ©rentiation 'f'{unite}', fontsize=12)
    plt.ylabel('Erreur ' f'{type}' '(mol/m^3)', fontsize=12)

    # Afficher l'Ã©quation de la rÃ©gression en loi de puissance
    equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Î”x^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')
    
    # DÃ©placer la zone de texte
    equation_text_obj.set_position((0.5, 0.4))
    
    plt.grid(True)
    plt.show()

    return None
    
    
    

# Recueil des erreurs
h_values, L1_N = reading_files_N("erreurs_L1_N")
t_values, L1_T = reading_files_T("erreurs_L1_T")
h_values, L2_N = reading_files_N("erreurs_L2_N")
t_values, L2_T = reading_files_T("erreurs_L2_T")
h_values, Linf_N = reading_files_N("erreurs_Linf_N")
t_values, Linf_T = reading_files_T("erreurs_Linf_T")

plt_L1_N = graph_convergence(h_values, L1_N, "L1", "la taille du maillage spatiale", "m")
plt_L1_T = graph_convergence(t_values, L1_T, "L1", "la taille du maillage temporel", "m")
plt_L2_N = graph_convergence(h_values, L2_N, "L2", "la taille du maillage spatiale", "m")
plt_L2_T = graph_convergence(t_values, L2_T, "L2", "la taille du maillage temporel", "m")
plt_Linf_N = graph_convergence(h_values, Linf_N, "Linf", "la taille du maillage spatiale", "m")
plt_Linf_T = graph_convergence(t_values, Linf_T, "Linf", "la taille du maillage temporel", "m")