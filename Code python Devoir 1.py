# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def make_A_matrix_order1(N, R):
    "Génération de la matrice A"
    dr = R/(N-1)
    A = np.zeros((N, N))
    #A[0,0] = -1/dr
    #A[0,1] = 1/dr
    A[0,0:2] = [-1/dr, 1/dr]
    
    for i in range(1, N-1):
        r = dr*i
        A[i, i-1:i+2] = [1/(dr**2), (-1/dr)*(1/r + 2/dr), (1/dr)*(1/r + 1/dr)]
    
    A[N-1, N-1] = 1
    
    return A

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

def make_b_vector(N, S, D_eff, C_e):
    "Génération de la matrice b"
    b = (S/D_eff)*np.ones(N)
    b[0] = 0
    b[N-1] = C_e
    
    return b

# Constantes
R = 0.5
N_start = 5
N_end = 10

S = 2*10**(-8)
C_e = 20
D_eff = 10**(-10)
vect_N= []
L1_1 = []
L1_2 = []
L2_1 = []
L2_2 = []
Linf_1 = []
Linf_2 = []

for N in range(N_start, N_end) :
    vect_N.append(R/N)

    A_matrix_1 = make_A_matrix_order1(N, 0.5)
    A_matrix_2 = make_A_matrix_order2(N, 0.5)

    b_vector = make_b_vector(N, S, D_eff, C_e)

    C_results_1 = np.linalg.solve(A_matrix_1, b_vector)
    C_results_2 = np.linalg.solve(A_matrix_2, b_vector)

    # Solution analytique
    r = np.linspace(0, R, N)
    C_exact = (1/4)*(S/D_eff)*(R**2)*((r**2)/(R**2) - 1) + C_e


    # Erreur totale commise
    #L1_1.append(np.trapz(C_results_1 - C_exact, r))
    #L1_2.append(np.trapz(C_results_2 - C_exact, r))

    #L2_1.append(np.sqrt(np.trapz((C_results_1 - C_exact)**2, r)))
    #L2_2.append(np.sqrt(np.trapz((C_results_2 - C_exact)**2, r)))

    #Linf_1.append((abs(C_results_1[0]-C_exact[0])))
    #Linf_2.append((abs(C_results_1[0]-C_exact[0])))

    # Erreur totale commise
    L1_1.append((1/N)*np.sum(np.abs(C_results_1 - C_exact)))
    L1_2.append((1/N)*np.sum(np.abs(C_results_2 - C_exact)))

    L2_1.append(np.sqrt((1/N)*np.sum(np.abs(C_results_1 - C_exact)**2)))
    L2_2.append(np.sqrt((1/N)*np.sum(np.abs(C_results_2 - C_exact)**2)))

    Linf_1.append(np.max(abs(C_results_1-C_exact)))
    Linf_2.append(np.max(abs(C_results_1-C_exact)))
   

print(vect_N)

# Affichage des résultats
plt.figure
plt.plot(r, C_exact, label="Exacte")
plt.plot(r, C_results_1, label="Numérique ordre1")
plt.title("E) Comparaison entre la solution numérique par différences finies\n et la solution exacte")
plt.legend()
plt.show()

plt.figure
plt.loglog(vect_N, L1_1, label="Erreur L1")
plt.loglog(vect_N, L2_1, label="Erreur L2")
plt.loglog(vect_N, Linf_1, label="Erreur Linf")
plt.title("Comparaison des erreurs L1, L2 et Linf pour la résolution avec schéma d'ordre 1")
plt.ylabel("Erreur (mol/m^3)")
plt.xlabel("Pas de différentiation (m)")
plt.legend()
plt.show()

plt.figure
plt.plot(r, C_exact, label="Exacte")
plt.plot(r, C_results_2, label="Numérique ordre 2")
plt.title("E) Comparaison entre la solution numérique par différences finies\n et la solution exacte")
plt.legend()
plt.show()

plt.figure
plt.loglog(vect_N, L1_2, label="Erreur L1")
plt.loglog(vect_N, L2_2, label="Erreur L2")
plt.loglog(vect_N, Linf_2, label="Erreur Linf")
plt.title("Comparaison des erreurs L1, L2 et Linf pour la résolution avec schéma d'ordre 2")
plt.ylabel("Erreur (mol/m^3)")
plt.xlabel("Pas de différentiation (m)")
plt.legend()
plt.show()

# Calcul de la convergence du système en prenant N et 2N
p_hat = (np.log(L1_1[0]/L1_1[4])/np.log(2))
print("Convergence L1_1 : ", np.log(L1_1[0]/L1_1[4])/np.log(2))

print("Convergence L1_2 : ", np.log(L1_2[0]/L1_2[4])/np.log(2))

# ----- Graphiques de convergences L_1_1
coefficients = np.polyfit(np.log(vect_N), np.log(L1_1), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, L1_1, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 1\n de l\'erreur $L_1$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_1$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_1 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# ----- Graphiques de convergences L_2_1
coefficients = np.polyfit(np.log(vect_N), np.log(L2_1), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, L2_1, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 1\n de l\'erreur $L_2$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_2$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# ----- Graphiques de convergences L_1_2
coefficients = np.polyfit(np.log(vect_N), np.log(L1_2), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, L1_2, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 1\n de l\'erreur $L_1$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_1$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_1 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# ----- Graphiques de convergences L_2_1
coefficients = np.polyfit(np.log(vect_N), np.log(L1_2), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, L1_2, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 2\n de l\'erreur $L_1$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_1$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_1 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))


# ----- Graphiques de convergences L_2_2
coefficients = np.polyfit(np.log(vect_N), np.log(L2_2), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, L2_2, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_2$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))



# ----- Graphiques de convergences Linf_1
coefficients = np.polyfit(np.log(vect_N), np.log(Linf_1), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, Linf_1, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 1\n de l\'erreur $L_inf$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_inf$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_inf = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))



# ----- Graphiques de convergences Linf_2
coefficients = np.polyfit(np.log(vect_N), np.log(Linf_2), 1)
exponent = coefficients[0]

# Fonction de régression en termes de logarithmes
fit_function_log = lambda x: exponent * x + coefficients[1]

# Fonction de régression en termes originaux
fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

# Extrapoler la valeur prédite pour la dernière valeur de h_values
extrapolated_value = fit_function(vect_N[-1])

# Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
plt.figure(figsize=(8, 6))
plt.scatter(vect_N, Linf_2, marker='o', color='b', label='Données numériques obtenues')
plt.plot(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')

# Marquer la valeur extrapolée
#plt.scatter(h_values[-1], extrapolated_value, marker='x', color='g', label='Extrapolation')

# Ajouter des étiquettes et un titre au graphique
plt.title('Convergence d\'ordre 2\n de l\'erreur $L_inf$ en fonction de $Δx$',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

plt.xlabel('Taille de maille $h_{max}$ ou $Δx$ (cm)', fontsize=12, fontweight='bold')  # Remplacer "h" par "Δx"
plt.ylabel('Erreur $L_inf$ (m/s)', fontsize=12, fontweight='bold')

# Afficher l'équation de la régression en loi de puissance
equation_text = f'$L_inf = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')

# Déplacer la zone de texte
equation_text_obj.set_position((0.5, 0.4))
