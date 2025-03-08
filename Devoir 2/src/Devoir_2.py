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
 
    for time in time_vector :
        b_S = copy.deepcopy(b) + dt*compute_M_st(r, time, Deff, k, R, Ce)
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

    for time in time_vector :
        b[0] = 0
        b[N-1] = 20
        C_t_pdt = copy.deepcopy(np.linalg.solve(A, b))
        #print("C_t_pdt = ", C_t_pdt)
        results_matrix.append(copy.deepcopy(C_t_pdt))
        #print("Results_matrix = ", results_matrix)
        b = copy.deepcopy(C_t_pdt)
        
    return results_matrix



# ----- Entr�e des donn�es -----
print("Current working directory:", os.getcwd())
start_delimiter = "START"
end_delimiter = "END"

data_dict = {}

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
r = np.linspace(0, R, N)

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
results_test = verification(N, R, dt, time_vector, k, Deff, Ce)

# Plot the results
r_values = np.linspace(0, R, N)
plt.figure(figsize=(10, 6))
for t in range(0, len(time_vector)):  # Plot every 5th time step
    plt.plot(r_values, results_test[t], label=f"t = {time_vector[t]:.1e} s")

plt.ylim(0, None)
plt.xlabel("r (m)")
plt.ylabel("C(r, t)")
plt.title("Numerical Solution at Different Time Steps")
plt.legend()
plt.grid(True)

file_path = os.path.join(output_folder, 'sol_manufacturee_num.png')
plt.savefig(file_path)
plt.show()




# ----- Solution manufactur�e -----
# Plot the manufactured solution
r_values = np.linspace(0, R, N)  # r from 0 to R
t_values = np.linspace(start, stop, Ntemps)  # Time from 0 to 4e9 seconds, 5 time steps

C_values = []

plt.figure(figsize=(10, 6))
for t in range(len(t_values)):
    C_values.append(C_MMS(r_values, t_values[t]))
    plt.plot(r_values, C_values[t], label=f"t = {t_values[t]:.1e} s")

plt.ylim(0, None)
plt.xlabel("r (m)")
plt.ylabel("C_MMS(r, t)")
plt.title("Manufactured Solution $C_{MMS}(r, t)$ for Different Time Steps")
plt.legend()
plt.grid(True)

file_path = os.path.join(output_folder, 'sol_manufacturee.png')
plt.savefig(file_path)
plt.show()




# ----- Solution num�rique de l'�nonc� -----
# Calcul de la solution du probl�me de l'�nonc�
results = resolution(N, R, dt, time_vector, k, Deff, Ce)

# Plot the results


# Define colormap and normalization
cmap = cm.viridis  
norm = plt.Normalize(vmin=0, vmax=Ntemps-1)

plt.figure(figsize=(10, 6))
for i in range(Ntemps):
    plt.plot(r_values, results[i], color=cmap(norm(i)), alpha=0.8)  # Gradient color

plt.ylim(0, None)
plt.xlabel("r (m)")
plt.ylabel("C(r, t)")
plt.title("Concentration dans le pilier en fonction de la position et du temps", fontweight='bold')

# Ensure colorbar is correctly linked
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=plt.gca())  # Explicitly associate with the current Axes
cbar.set_label("Time step (t)")

file_path = os.path.join(output_folder, 'sol_pilier_num.png')
plt.savefig(file_path)
plt.show()



