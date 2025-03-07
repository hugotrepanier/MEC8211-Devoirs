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
    A[0, 0] = 1        # Condition de Dirichlett à la frontière
    A[N-1,N-3:N] = [1, -4, 3]

# terme source manufacturé à implanter dans ce code pour trouver la solution manufacturée
    M_st = -4*Ce*Deff*np.exp(-k*t)/R**2 + Ce*k*(1 - r**2/R**2)*np.exp(-k*t) - k*(-Ce*(1 - r**2/R**2)*np.exp(-k*t) + Ce)

    for i in range(1, N-1):
        r = dr*i
        A[i, i-1:i+2] = [((dt*Deff)/(2*r*dr) - (dt*Deff)/(dr**2)),
                         (1 + dt*k + (2*dt*Deff)/(dr**2)),
                         ((-dt*Deff)/(2*r*dr) - (dt*Deff)/(dr**2))]
    return A

def make_b_vector(N, Deff, Ce):
    "Génération de la matrice b"
    b = np.zeros(N)
    b[0] = Ce
    
    return b



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
dt = (stop-start)/Ntemps

results_matrix = []

A = make_A_matrix_order2(N, R, dt, k, Deff,Ce)
b = make_b_vector(N, Deff, Ce)
results_matrix.append(b)

for t in time_vector :
    b[N-1] = 0
    C_t_pdt = np.linalg.solve(A, b)
    b = C_t_pdt
    results_matrix.append(b)

# Define colormap and normalization
cmap = cm.viridis  
norm = plt.Normalize(vmin=0, vmax=Ntemps-1)

plt.figure(figsize=(8, 5))
for i in range(Ntemps):
    plt.plot(r, results_matrix[i][::-1], color=cmap(norm(i)), alpha=0.8)  # Gradient color

plt.title("Concentration dans le pilier en fonction de la position et du temps", fontweight='bold')
plt.ylabel("Concentration (mol/m^3)")
plt.xlabel("Position (m)")

# Ensure colorbar is correctly linked
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=plt.gca())  # Explicitly associate with the current Axes
cbar.set_label("Time step (t)")

plt.show()

