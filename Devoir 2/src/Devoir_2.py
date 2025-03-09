# -*- coding: utf-8 -*-
"""
POLYTECHNIQUE MONTRÃ‰AL
MEC8211 : VÃ‰RIFICATION ET VALIDATION EN MODÃ‰LISATION NUMÃ‰RIQUE

DEVOIR 2 : VÃ‰RIFICATION DE CODE PAR LA MÃ‰THODE DES SOLUTIONS MANUFACTURÃ‰ES

PAR : Hugo TrÃ©panier, Jean Zhu & Renno Soussa
"""

"""
Ce code permet de rÃ©soudre l'Ã©quation de diffusion-rÃ©action transitoire impliquÃ©e dans le problÃ¨me de diffusion du sel minÃ©ral dissout dans l'eau
Ã  travers un pilier de bÃ©ton en contact avec l'eau saline.

L'objectif est d'obtenir le profil de concentration en coordonnÃ©es cylindrique selon la position dans le rayon du pilier de bÃ©ton, et ce, pour 
plusieurs pas de temps. Cela permet alors de caractÃ©riser l'Ã©volution temporelle du transfert de masse dans le pilier.

La rÃ©solution spatiale se fait par la mÃ©thode des diffÃ©rences finies avec des schÃ©mas de diffÃ©rentiation d'ordre 2 pour permettre une rÃ©solution exacte
de l'Ã©quation diffÃ©rentielle spatiale, et ce, puisque la solution analytique est un polynÃ´me d'ordre 2.
La rÃ©solution temporelle se fait par la mÃ©thode de XXXXXXXXXXXXXXXXXX puisque XXXXXXXXXXXXXXXXXXX

Ã€ la fin de la rÃ©soltion, nous nous attendons Ã  obtenir un graphique du profil de concentration selon la position dans le rayon pour Ã  plusieurs temps t.
Le profil parabolique sera plus prononcÃ© aux premiers pas de temps et et s'aplatira au fil du temps. Une rÃ©action d'ordre 1 consommant le sel dans le pilier
existe, donc, le profil parabolique ne deviendra probablement pas complÃ¨tement plat puisqu'un gradient de concentration se maintiendra.

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
Fontions utilisÃ©es dans la rÃ©solution de l'Ã©quation diffÃ©rentielle
"""
# CrÃ©ation de la matrice de discrÃ©tisation spatiale de l'e.d.p.
def make_A_matrix_order2(N, R, dt, k, Deff, Ce):
    "GÃ©nÃ©ration de la matrice A"
    dr = R/(N-1)
    A = np.zeros((N, N))
            # Condition de Dirichlett Ã  la frontiÃ¨re
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
    "GÃ©nÃ©ration de la matrice b"
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
    
    # DiscrÃ©tisation dans l'espace
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
    
    # DiscrÃ©tisation dans l'espace
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
    " Calcul de l'erreur L1 pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_N = ((1/N)*np.sum(np.abs(results_num_np[Ntemps-1] - results_analytique_np[Ntemps-1])))

    return L1_N

def calcul_L1_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L1 pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L1_N = ((1/Ntemps)*np.sum(np.abs(results_num_np[:,0] - results_analytique_np[:,0])))

    return L1_N

def calcul_L2_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_N = (np.sqrt((1/N)*np.sum(np.abs(results_num_np[Ntemps-1] - results_analytique_np[Ntemps-1])**2)))

    return L2_N

def calcul_L2_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur L2 pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    L2_T = (np.sqrt((1/Ntemps)*np.sum(np.abs(results_num_np[:,0] - results_analytique_np[:,0])**2)))

    return L2_T

def calcul_Linf_N(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'étude de l'influence de N. L'erreur est calculé en utilisant la dernière courbe obtenue."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_N = (np.max(abs(results_num_np[Ntemps-1]-results_analytique_np[Ntemps-1])))

    return Linf_N

def calcul_Linf_T(N, Ntemps, results_num, results_analytique) :
    " Calcul de l'erreur Linf pour l'étude de l'influence de T. L'erreur est calculé en utilisant toutes les courbes obtenues à la position centrale."
    results_num_np = np.array(results_num)
    results_analytique_np = np.array(results_analytique)
    Linf_T = (np.max(abs(results_num_np[:,0]-results_analytique_np[:,0])))

    return Linf_T




# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python Devoir_2.py <path_to_Input_data.txt>")
    sys.exit(1)

# Get the path to Input_data.txt from the command-line argument
input_file_path = sys.argv[1]

# Debug: Print the input file path
print("Reading input file from:", input_file_path)

# ----- Entrée des données -----
start_delimiter = "START"
end_delimiter = "END"

data_dict = {}

with open(input_file_path, "r") as file:
    capture = False
    for line in file:
        line = line.strip()
        if line == start_delimiter:
            capture = True
            continue  # Skip the delimiter line itself
        elif line == end_delimiter:
            break  # Stop reading when reaching END

        if capture:
            # Match variable assignments (e.g., "k = 4*10**(-9)" or "plot = Affirmative")
            match = re.match(r"(\w+)\s*=\s*(.+)", line)
            if match:
                key, value = match.groups()
                # Evaluate numerical expressions, but treat strings as-is
                if key == "plot":
                    data_dict[key] = value.strip('"')  # Remove quotes if present
                else:
                    try:
                        data_dict[key] = eval(value)
                    except Exception as e:
                        print(f"Error parsing {key}: {value} -> {e}")
            else:
                print(f"Skipping line (no match): {line}")

# Debug: Print the parsed data
print("Parsed data:", data_dict)

# Extract variables
N = data_dict.get("N")
R = data_dict.get("R")
Ce = data_dict.get("Ce")
Ntemps = data_dict.get("Ntemps")
start = data_dict.get("temps_start")
stop = data_dict.get("temps_stop")
k = data_dict.get("k")
Deff = data_dict.get("Deff")
plot = data_dict.get("plot")  # Get the plot flag

# Check if N and R are valid
if N is None or R is None:
    raise ValueError("N or R is None. Check the input file.")

# Paramï¿½tres spatiaux      
r_values = np.linspace(0, R, N)

# Paramï¿½tres temporels
time_vector = np.linspace(start, stop, Ntemps)
dt = time_vector[1] - time_vector[0]

# Chemin pour la sauvegarde des graphiques
output_folder = '../results'

# ----- Vï¿½rification du code par la mï¿½thode des solutions manufacturï¿½es (MMS) -----
# Call the verification function
results_verif_num = verification(N, R, dt, time_vector, k, Deff, Ce)

# # ----- Solution manufacturï¿½e -----
# # Plot the manufactured solution
C_manuf = []
for t in range(len(time_vector)):
     C_manuf.append(C_MMS(r_values, time_vector[t]))

# # ----- Solution numï¿½rique de l'ï¿½noncï¿½ -----
# # Calcul de la solution du problï¿½me de l'ï¿½noncï¿½
results_pilier_num = resolution(N, R, dt, time_vector, k, Deff, Ce)

if plot=="Affirmative" :
  # Plot the results
  plt1 = plot_vs_rt(r_values, time_vector, results_verif_num, 
                    "Numerical Solution at Different Time Steps", 'sol_manufacturee_num.png')
  # Plot the results
  plt2 = plot_vs_rt(r_values, time_vector, C_manuf, 
                     "Manufactured Solution $C_{MMS}(r, t)$ for Different Time Steps", 'sol_manufacturee.png')
  
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

