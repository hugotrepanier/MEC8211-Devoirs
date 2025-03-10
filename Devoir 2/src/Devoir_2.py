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
import sys
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
    result = -4*Ce*Deff*np.exp(-k*t)/R**2 + Ce*k*(1 - r**2/R**2)*np.exp(-k*t) + k*(-Ce*(1 - r**2/R**2)*np.exp(-k*t) + Ce)
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
    Génération, affichage et sauvegarde des graphiques des résultats
    
    Paramètres
    ----------
    r_vetor (array) : Vecteur des points de discrétisation dans l'espace
    t_vector (array) : Vecteur des points de discrétisation dans le temps
    C_matrix (array (len(t_vector), len(r_vector))) : Matrice des résultats
    title (string) : Titre du graphique à afficher
    output_name (string) : Nom du fichier de sauvegarde

    Sorties
    -------
    File : Sauvegarde du graphique
    """
    # Definiton de la palette de couleurs
    cmap = cm.viridis  
    norm = plt.Normalize(vmin=0, vmax=Ntemps-1)
    
    # Affichage des resultats
    plt.figure(figsize=(10, 6))
    for t in range(len(t_vector)):
        plt.plot(r_vector, C_matrix[t], color=cmap(norm(t)), alpha=0.8, label=f"t = {time_vector[t]:.1e} s")

    plt.ylim(0, None)
    plt.xlabel("Position sur le rayon (m)")
    plt.ylabel("Concentration dans le temps (mol/m^3)")
    plt.title(title)


    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Atteinte du régime stationnaire (%)")
    plt.grid(True)
    file_path = os.path.join(output_folder, output_name)
    plt.savefig(file_path)
    plt.show()
    
    return None

def calcul_L1_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L1 pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_N = ((1/N)*(1/Ntemps)*np.sum(np.abs(results_num_np - results_analytique_np)))

    return L1_N

def calcul_L1_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L1 pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_T = ((1/N)*(1/Ntemps)*np.sum(np.abs(results_num_np - results_analytique_np)))

    return L1_T

def calcul_L2_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_N = (np.sqrt((1/N)*(1/Ntemps)*np.sum(np.abs(results_num_np - results_analytique_np)**2)))

    return L2_N

def calcul_L2_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_T = (np.sqrt((1/N)*(1/Ntemps)*np.sum(np.abs(results_num_np - results_analytique_np)**2)))

    return L2_T

def calcul_Linf_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_N = (np.max(abs(results_num_np-results_analytique_np)))

    return Linf_N

def calcul_Linf_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_T = (np.max(abs(results_num_np-results_analytique_np)))

    return Linf_T

def graph_convergence(maillages, erreurs, type, dimension, unité):
    coefficients = np.polyfit(np.log(maillages), np.log(erreurs), 1)
    exponent = coefficients[0]

    # Fonction de régression en termes de logarithmes
    fit_function_log = lambda x: exponent * x + coefficients[1]

    # Fonction de régression en termes originaux
    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

    # Extrapoler la valeur prédite pour la dernière valeur de h_values
    extrapolated_value = fit_function(maillages[-1])

    # Tracer le graphique en échelle log-log avec des points et la courbe de régression extrapolée
    plt.figure(figsize=(8, 6))
    plt.scatter(maillages, erreurs, marker='o', color='b', label='Données numériques obtenues')
    plt.loglog(maillages, fit_function(maillages), linestyle='--', color='r', label='Régression en loi de puissance')
    
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





# ----- Entree des donnees -----
# Verifier si le bon nombre d'argument est fourni
if len(sys.argv) != 2:
    print("Usage: python Devoir_2.py <path_to_Input_data.txt>")
    sys.exit(1)

# Recuperer l'addresse du fichier Input_data.txt de l'argument
input_file_path = sys.argv[1]

# Impression de l'addresse du fichier Input_data.txt
print("Reading input file from:", input_file_path)

# Lecture des donnees du fichier Input_data.txt
start_delimiter = "START"
end_delimiter = "END"

data_dict = {}

with open(input_file_path, "r") as file:
    capture = False
    for line in file:
        line = line.strip()
        if line == start_delimiter:
            capture = True
            continue  
        elif line == end_delimiter:
            break  

        if capture:
            
            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if match:
                key, value = match.groups()
                
                if key == "plot":
                    data_dict[key] = value.strip('"')
                else:
                    try:
                        data_dict[key] = eval(value)
                    except Exception as e:
                        print(f"Error parsing {key}: {value} -> {e}")
            else:
                print(f"Skipping line (no match): {line}")

# Impression des donnees recuperees
print("Parsed data:", data_dict)

# Extraction des variables
N = data_dict.get("N")
R = data_dict.get("R")
Ce = data_dict.get("Ce")
Ntemps = data_dict.get("Ntemps")
start = data_dict.get("temps_start")
stop = data_dict.get("temps_stop")
k = data_dict.get("k")
Deff = data_dict.get("Deff")
plot = data_dict.get("plot")

# Verifier si N et R sont valides
if N is None or R is None:
    raise ValueError("N or R is None. Check the input file. Verify if there is an extra line after the last value in liste_des_resolutions or liste_des_pas_de_temps.")

# Conversion a une matrice
matrix = [[data_dict.get("k", 0), data_dict.get("Deff", 0), 
           data_dict.get("Ce", 0), data_dict.get("N", 0), 
           data_dict.get("Ntemps", 0)]]

# Discretisation spatiale    
r_values = np.linspace(0, R, N)

# Discretisation temporelle
time_vector = np.linspace(start, stop, Ntemps)
dt = time_vector[1] - time_vector[0]

# Chemin pour la sauvegarde des graphiques
output_folder = '../results'

# ----- Verification du code par la methode des solutions manufacturees (MMS) -----
# Appel de la fonction de verification
results_verif_num = verification(N, R, dt, time_vector, k, Deff, Ce)

# # ----- Solution manufacturee -----
C_manuf = []
for t in range(len(time_vector)):
     C_manuf.append(C_MMS(r_values, time_vector[t]))

# # ----- Solution numerique de l'enonce -----
# # Calcul de la solution du probleme de l'enonce
results_pilier_num = resolution(N, R, dt, time_vector, k, Deff, Ce)

# Affichage des graphiques
if plot=="Affirmative" :
    # Affichage du graphique des resultats numeriques obtenus pour la solution manufacturee
    plt1 = plot_vs_rt(r_values, time_vector, results_verif_num, 
                    "Numerical Solution at Different Time Steps", 'sol_manufacturee_num.png')

    # Affichage du graphique des valeurs analytiques de la solution manufacturee
    plt2 = plot_vs_rt(r_values, time_vector, C_manuf, 
                     "Manufactured Solution $C_{MMS}(r, t)$ for Different Time Steps", 'sol_manufacturee.png')
  
    # Affichage du graphiques des resultats numeriques du probleme de l'enonce
    plt1 = plot_vs_rt(r_values, time_vector, results_pilier_num, 
                    "Concentration dans le pilier en fonction de la position et du temps", 'sol_pilier_num.png')


# Calcul des erreurs entre la valeur numerique et la valeur analytique de la solution manufacturee
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












