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
import matplotlib.cm as cm
import copy
import re
import os

"""
Fontions utilisées dans la résolution de l'équation différentielle
"""
# Création de la matrice de discrétisation spatiale de l'e.d.p.
def make_A_matrix_order2(N, R, dt, k, Deff, Ce):
    "Génération de la matrice A"
    dr = R/(N-1)
    A = np.zeros((N, N))
            # Condition de Dirichlett à la frontière
    # A[N-1,N-3:N] = [1, -4, 3]
    A[0,0:3] = [-3, 4, -1]
    A[N-1, N-1] = 1

    for i in range(1, N-1):
        r = dr*i
        A[i, i-1:i+2] = [((dt*Deff)/(2*r*dr) - (dt*Deff)/(dr**2)),
                         (1 + k*dt + (2*dt*Deff)/(dr**2)),
                         ((-dt*Deff)/(2*r*dr) - (dt*Deff)/(dr**2))]
    return A

def make_b_vector(N, Deff, Ce):
    "Génération de la matrice b"
    b = np.zeros(N)
    
    return b

def C_MMS(r, t):
    return -Ce* (1 - r**2 / R**2) * np.exp(-4 * 1e-9 * t) + Ce

def compute_M_st(r, t, Deff, k, R, Ce):
    # term_1 = (80*Deff*np.exp((-4.0*10**(-9)) * t))/(R**2)
    # term_2 = k*((-20 + 20*(r**2)/(R**2))*np.exp((-4*10**(-9)) * t) + 20)
    # term_3 = (-4.0*10**(-9))*(-20 + (20*r**2)/(R**2))*np.exp((-4*10**(-9))*t)
    # result = term_1 + term_2 + term_3
    result = -4*Ce*Deff*np.exp(-k*t)/R**2 + Ce*k*(1 - r**2/R**2)*np.exp(-k*t) + k*(-Ce*(1 - r**2/R**2)*np.exp(-k*t) + Ce)
    # result = (-4*Deff*Ce*np.exp(-k*t))/(R**2) + (k*Ce)
    return result

def verification(N, R, dt, t_vector, k, Deff, Ce) :
    
    # Discrétisation dans l'espace
    r_vector = np.linspace(0, R, N)
    dr = R/(N-1)
    
    A = make_A_matrix_order2(N, R, dt, k, Deff,Ce)
    b = make_b_vector(N, Deff, Ce)
    b = copy.deepcopy(b) + C_MMS(r_vector, 0)
 
    results_matrix = [b]
 
    for time in time_vector[1:] :
        b_S = copy.deepcopy(b) + dt*compute_M_st(r_vector, time, Deff, k, R, Ce)
        b_S[0] = 0
        b_S[N-1] = 20
        C_t_pdt = copy.deepcopy(np.linalg.solve(A, b_S))
        results_matrix.append(copy.deepcopy(C_t_pdt))
        b = copy.deepcopy(C_t_pdt)
    
    return results_matrix

def resolution(N, R, dt, t_vector, k, Deff, Ce) :
    
    # Discrétisation dans l'espace
    r_vector = np.linspace(0, R, N)
    dr = R/(N-1)
    
    results_matrix = []
    A = make_A_matrix_order2(N, R, dt, k, Deff,Ce)
    b = make_b_vector(N, Deff, Ce)

    results_matrix.append(b)

    for time in time_vector[1:] :
        b[0] = 0
        b[N-1] = 20
        C_t_pdt = copy.deepcopy(np.linalg.solve(A, b))
        #print("C_t_pdt = ", C_t_pdt)
        results_matrix.append(copy.deepcopy(C_t_pdt))
        #print("Results_matrix = ", results_matrix)
        b = copy.deepcopy(C_t_pdt)
        
    return results_matrix

def plot_vs_rt(r_vector, t_vector, C_matrix, title, output_name) :
    """
    G�n�ration, affichage et sauvegarde des graphiques des r�sultats
    
    Param�tres
    ----------
    r_vetor (array) : Vecteur des points de discr�tisation dans l'espace
    t_vector (array) : Vecteur des points de discr�tisation dans le temps
    C_matrix (array (len(t_vector), len(r_vector))) : Matrice des r�sultats
    title (string) : Titre du graphique � afficher
    output_name (string) : Nom du fichier de sauvegarde

    Sorties
    -------
    File : Sauvegarde du graphique
    """
    # Define colormap and normalization
    cmap = cm.viridis  
    norm = plt.Normalize(vmin=0, vmax=Ntemps-1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    for t in range(len(t_vector)):  # Plot every 5th time step
        plt.plot(r_vector, C_matrix[t], color=cmap(norm(t)), alpha=0.8, label=f"t = {time_vector[t]:.1e} s")

    plt.ylim(0, None)
    plt.xlabel("r (m)")
    plt.ylabel("C(r, t)")
    plt.title(title)

    # Ensure colorbar is correctly linked
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=plt.gca())  # Explicitly associate with the current Axes
    cbar.set_label("Time step (t)")
    plt.grid(True)
    file_path = os.path.join(output_folder, output_name)
    plt.savefig(file_path)
    plt.show()
    
    return None

def calcul_L1_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L1 pour l'�tude de l'influence de N. L'erreur est calcul� en utilisant la derni�re courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_N = ((1/N)*np.sum(np.abs(results_num_np[Ntemps-1] - results_analytique_np[Ntemps-1])))

    return L1_N

def calcul_L1_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L1 pour l'�tude de l'influence de T. L'erreur est calcul� en utilisant toutes les courbes obtenues � la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_T = ((1/Ntemps)*np.sum(np.abs(results_num_np[:,0] - results_analytique_np[:,0])))

    return L1_T

def calcul_L2_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'�tude de l'influence de N. L'erreur est calcul� en utilisant la derni�re courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_N = (np.sqrt((1/N)*np.sum(np.abs(results_num_np[Ntemps-1] - results_analytique_np[Ntemps-1])**2)))

    return L2_N

def calcul_L2_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'�tude de l'influence de T. L'erreur est calcul� en utilisant toutes les courbes obtenues � la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_T = (np.sqrt((1/Ntemps)*np.sum(np.abs(results_num_np[:,0] - results_analytique_np[:,0])**2)))

    return L2_T

def calcul_Linf_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'�tude de l'influence de N. L'erreur est calcul� en utilisant la derni�re courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_N = (np.max(abs(results_num_np[Ntemps-1]-results_analytique_np[Ntemps-1])))

    return Linf_N

def calcul_Linf_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'�tude de l'influence de T. L'erreur est calcul� en utilisant toutes les courbes obtenues � la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_T = (np.max(abs(results_num_np[:,0]-results_analytique_np[:,0])))

    return Linf_T

def graph_convergence(maillages, erreurs, type, dimension, unité):
    coefficients = np.polyfit(np.log(maillages), np.log(erreurs), 1)
    exponent = coefficients[0]

    # Fonction de régression en termes de logarithmes
    fit_function_log = lambda x: exponent * x + coefficients[1]

    # Fonction de régression en termes originaux
    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

    # Extrapoler la valeur prédite pour la dernière valeur de h_values
    extrapolated_value = fit_function(vect_N[-1])

    # Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
    plt.figure(figsize=(8, 6))
    plt.scatter(vect_N, erreurs, marker='o', color='b', label='Données numériques obtenues')
    plt.loglog(vect_N, fit_function(vect_N), linestyle='--', color='r', label='Régression en loi de puissance')
    
    # Ajouter des étiquettes et un titre au graphique
    plt.title('Convergence d\'ordre 1\n de l\'erreur'f'{type}'  'en fonction de' f'{dimension}',
          fontsize=14, fontweight='bold', y=1.02)  # Le paramètre y règle la position verticale du titre

    plt.xlabel('Pas de différentiation'f'{unité}', fontsize=12)
    plt.ylabel('Erreur $L_2$ (mol/m^3)', fontsize=12)

    # Afficher l'équation de la régression en loi de puissance
    equation_text = f'$L_2 = {np.exp(coefficients[1]):.4f} \\times Δx^{{{exponent:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=12, transform=plt.gca().transAxes, color='k')
    
    # Déplacer la zone de texte
    equation_text_obj.set_position((0.5, 0.4))
    
    plt.grid(True)
    plt.show()


    return None





# ----- Entr�e des donn�es -----
start_delimiter = "START"
end_delimiter = "END"

data_dict = {}

# Chemin vers le fichier de données d'entrée
with open("../Devoir 2/data/Input_data.txt", "r") as file:
    capture = False
    for line in file:
        line = line.strip()
        if line == start_delimiter:
            capture = True
            continue  # Skip the delimiter line itself
        elif line == end_delimiter:
            break  # Stop reading when reaching END

        if capture:
            # Match variable assignments (e.g., "k = 4*10**(-9)")
            match = re.match(r"(\w+)\s*=\s*([\d\*\(\)\-\+\./eE]+)", line)
            if match:
                key, value = match.groups()
                # Evaluate the expression safely
                try:
                    data_dict[key] = eval(value)
                except Exception as e:
                    print(f"Error parsing {key}: {value} -> {e}")

# Convert to matrix (list of lists for numerical data only)
matrix = [[data_dict.get("k", 0), data_dict.get("Deff", 0), 
           data_dict.get("Ce", 0), data_dict.get("N", 0), 
           data_dict.get("Ntemps", 0)]]

# Param�tres spatiaux      
N = data_dict.get("N")
R = data_dict.get("R")
Ce = data_dict.get("Ce")
r_values = np.linspace(0, R, N)

# Param�tres temporels
Ntemps = data_dict.get("Ntemps")
start = data_dict.get("temps_start")
stop = data_dict.get("temps_stop")

# Param�tres physiques
k = data_dict.get("k")
Deff = data_dict.get("Deff")
time_vector = np.linspace(start, stop, Ntemps)
dt = time_vector[1] - time_vector[0]

# Chemin pour la sauvegarde des graphiques
output_folder = '../Devoir 2/results'

# ----- V�rification du code par la m�thode des solutions manufactur�es (MMS) -----
# Call the verification function
results_verif_num = verification(N, R, dt, time_vector, k, Deff, Ce)

# Plot the results
plt1 = plot_vs_rt(r_values, time_vector, results_verif_num, "Numerical Solution at Different Time Steps", 'sol_manufacturee_num.png')

# # ----- Solution manufactur�e -----
# # Plot the manufactured solution
C_manuf = []
for t in range(len(time_vector)):
     C_manuf.append(C_MMS(r_values, time_vector[t]))

# Plot the results
plt_2 = plot_vs_rt(r_values, time_vector, C_manuf, "Manufactured Solution $C_{MMS}(r, t)$ for Different Time Steps", 'sol_manufacturee.png')

# # ----- Solution num�rique de l'�nonc� -----
# # Calcul de la solution du probl�me de l'�nonc�
results_pilier_num = resolution(N, R, dt, time_vector, k, Deff, Ce)

# Plot the results
plt1 = plot_vs_rt(r_values, time_vector, results_pilier_num, 
                  "Concentration dans le pilier en fonction de la position et du temps", 'sol_pilier_num.png')

L1_N = calcul_L1_N(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur L1_N pour N =", N, " est :", L1_N)

L1_T = calcul_L1_T(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur L1_T pour Ntemps =", Ntemps, " est :", L1_T)

L2_N = calcul_L2_N(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur L2_N pour N =", N, " est :", L2_N)

L2_T = calcul_L2_T(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur L2_T pour Ntemps =", Ntemps, " est :", L2_T)

Linf_N = calcul_Linf_N(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur Linf_N pour N =", N, " est :", Linf_N)

Linf_T = calcul_Linf_T(N, Ntemps, results_verif_num, C_manuf)
print("L'erreur Linf_T pour Ntemps =", Ntemps, " est :", Linf_T)

"""
Code pour générer les graphiques de convergence à même le code principal

maillages = [10,50,250,1000]
vect_N= []
for N in maillages:

    r_values = np.linspace(0, R, N)
    time_vector = np.linspace(start, stop, Ntemps)
    solution_numérique = verification(N, R, dt, time_vector, k, Deff, Ce)
    solution_manufacturée = C_MMS(r_values, time_vector[0])
    print(solution_manufacturée)

    for t in range(0,len(time_vector)):

        solution_manufacturée[t] = C_MMS(r_values, t)

    print(len(solution_manufacturée))
    print(len(solution_numérique))

    L1_N = calcul_L1_N(N, Ntemps, solution_numérique, solution_manufacturée)
    L1_T = calcul_L1_T(N, Ntemps, solution_numérique, solution_manufacturée)
    L2_N = calcul_L2_N(N, Ntemps, solution_numérique, solution_manufacturée)
    L2_T = calcul_L2_T(N, Ntemps, solution_numérique, solution_manufacturée)
    Linf_N = calcul_Linf_N(N, Ntemps, solution_numérique, solution_manufacturée)
    Linf_T = calcul_Linf_T(N, Ntemps, solution_numérique, solution_manufacturée)

    vect_N.append(R/(N-1))

    print(N)
"""










