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

def graph_convergence(maillages, L1, L2, L8, dimension, unite, nom):
    coefficients1 = np.polyfit(np.log(maillages), np.log(L1), 1)
    exponent1 = coefficients1[0]
    coefficients2 = np.polyfit(np.log(maillages), np.log(L2), 1)
    exponent2 = coefficients2[0]
    coefficients8 = np.polyfit(np.log(maillages), np.log(L8), 1)
    exponent8 = coefficients8[0]

    # Fonction de rÃ©gression en termes de logarithmes
    fit_function_log1 = lambda x: exponent1 * x + coefficients1[1]
    fit_function_log2 = lambda x: exponent2 * x + coefficients2[1]
    fit_function_log8 = lambda x: exponent8 * x + coefficients8[1]

    # Fonction de rÃ©gression en termes originaux
    fit_function1 = lambda x: np.exp(fit_function_log1(np.log(x)))
    fit_function2 = lambda x: np.exp(fit_function_log2(np.log(x)))
    fit_function8 = lambda x: np.exp(fit_function_log8(np.log(x)))

    # Extrapoler la valeur prÃ©dite pour la derniÃ¨re valeur de h_values
    extrapolated_value1 = fit_function1(maillages[-1])
    extrapolated_value2 = fit_function2(maillages[-1])
    extrapolated_value8 = fit_function8(maillages[-1])

    # Tracer le graphique en Ã©chelle log-log avec des points et la courbe de rÃ©gression extrapolÃ©e
    plt.figure(figsize=(6, 6))
    plt.scatter(maillages, L1, marker='o', color='b', label='Données numériques obtenues pour L1')
    plt.loglog(maillages, fit_function1(maillages), linestyle='--', color='r', label='Régression en loi de puissance')
    plt.scatter(maillages, L2, marker='o', color='b', label='Données numériques obtenues pour L2')
    plt.loglog(maillages, fit_function2(maillages), linestyle='--', color='r', label='Régression en loi de puissance')
    plt.scatter(maillages, L8, marker='o', color='b', label='Données numériques obtenues pour Linfini')
    plt.loglog(maillages, fit_function8(maillages), linestyle='--', color='r', label='Régression en loi de puissance')
    
    # Ajouter des Ã©tiquettes et un titre au graphique
    plt.title('Normes des erreurs en fonction de ' f'{dimension}',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramÃ¨tre y rÃ¨gle la position verticale du titre

    plt.xlabel('Pas de différentiation 'f'({unite})', fontsize=12)
    plt.ylabel('Erreur''(mol/m^3)', fontsize=12)

    # Afficher les équations sur le graphique (sans boîte et un peu plus haut)
    equation_text1 = f'$L_1 = {np.exp(coefficients1[1]):.10f} \\times dx^{{{exponent1:.4f}}}$'
    equation_text2 = f'$L_2 = {np.exp(coefficients2[1]):.10f} \\times dx^{{{exponent2:.4f}}}$'
    equation_text8 = f'$L_\\infty = {np.exp(coefficients8[1]):.10f} \\times dx^{{{exponent8:.4f}}}$'

    # Positionner les équations près des courbes (un peu plus haut)
    y_offset = 0.9  # Facteur pour déplacer le texte vers le haut
    plt.text(maillages[1], L1[1] * y_offset, equation_text1, fontsize=12, color='r')
    plt.text(maillages[1], L2[1] * y_offset, equation_text2, fontsize=12, color='m')
    plt.text(maillages[1], L8[1] * y_offset, equation_text8, fontsize=12, color='c')
    
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

# Affichage des graphiques de convergence
plt_L_N = graph_convergence(h_values, L1_N, L2_N, Linf_N, "la taille du maillage spatial", "m", "plt_LN")
plt_L_T = graph_convergence(t_values, L1_T, L2_T, Linf_T, "la taille du maillage temporel", "s", "plt_LT")

"""
plt_L1_N = graph_convergence(h_values, L1_N, "L1", "la taille du maillage spatiale", "m")
plt_L1_T = graph_convergence(t_values, L1_T, "L1", "la taille du maillage temporel", "m")
plt_L2_N = graph_convergence(h_values, L2_N, "L2", "la taille du maillage spatiale", "m")
plt_L2_T = graph_convergence(t_values, L2_T, "L2", "la taille du maillage temporel", "m")
plt_Linf_N = graph_convergence(h_values, Linf_N, "Linf", "la taille du maillage spatiale", "m")
plt_Linf_T = graph_convergence(t_values, Linf_T, "Linf", "la taille du maillage temporel", "m")
"""
