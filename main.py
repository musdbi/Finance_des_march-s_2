import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Définition du chemin vers le dossier contenant les fichiers Excel
dossier_donnees = "./data_luxe"

# Chargement des fichiers Excel contenant les données des actifs
BD = [pd.read_excel(os.path.join(dossier_donnees, fichier), header=None,
                     names=["Actif", "Dates", "Ouverture", "Max", "Min", "Fermeture", "Volume"])
      for fichier in os.listdir(dossier_donnees) if fichier.endswith((".xlsm", ".xlsx"))]

# Vérification du nombre de fichiers
num_assets = len(BD)

# Initialisation de la matrice d'espérance de rendement ER
ER = np.zeros((num_assets, 1))

# Calcul de l'espérance de rendement pour chaque actif
for j in range(num_assets):
    bd = BD[j]
    St = [(bd.iloc[i, 3] + bd.iloc[i, 4]) / 2 for i in range(len(bd) - 1)]  # Moyenne (Max + Min) / 2
    Rt = [(St[i + 1] / St[i]) - 1 for i in range(len(St) - 1)]  # Rendement journalier ajusté
    ER[j] = np.mean(Rt)

# Initialisation de la matrice de covariance SIGMA
SIGMA = np.zeros((num_assets, num_assets))

# Calcul de la matrice de covariance
for i in range(num_assets):
    bdi = BD[i]
    Sti = [(bdi.iloc[k, 3] + bdi.iloc[k, 4]) / 2 for k in range(len(bdi) - 1)]
    Rti = np.array([(Sti[k + 1] / Sti[k]) - 1 for k in range(len(Sti) - 1)])

    for j in range(num_assets):
        bdj = BD[j]
        Stj = [(bdj.iloc[k, 3] + bdj.iloc[k, 4]) / 2 for k in range(len(bdj) - 1)]
        Rtj = np.array([(Stj[k + 1] / Stj[k]) - 1 for k in range(len(Stj) - 1)])

        SIGMA[i, j] = np.cov(Rti, Rtj)[0, 1]

# Calcul de l'inverse de la matrice de covariance
SIGMAINV = np.linalg.pinv(SIGMA)

# Calcul des coefficients a et b pour la droite de frontière efficiente
x = np.ones((10, 1))
xT = np.transpose(x)

a = np.dot(xT, np.dot(SIGMAINV, x)).item()
print("a:", a)

b = np.dot(xT, np.dot(SIGMAINV, ER)).item()
print("b:", b)

# Calcul du vecteur orthogonal
vect = ER - (b / a) * x
print("vecteur orthogonal:", vect)

# Norme de ce vecteur
norm2 = np.dot(np.transpose(vect), np.dot(SIGMAINV, vect)).item()
norm = np.sqrt(norm2)
print("norme:", norm)

# Fonctions pour la frontière efficiente
def FE(sig):
    return cste(sig) + np.sqrt(sig**2 - (1 / a)) * norm

def FNE(sig):
    return cste(sig) - np.sqrt(sig**2 - (1 / a)) * norm

def cste(sig):
    return b / a

# Simulation de portefeuilles aléatoires
NS = 100000  # Nombre de simulations
Axe1, Axe2 = [], []

for _ in range(NS):
    x = np.random.randn(10)
    sumx = np.sum(x)
    if sumx == 0 :
      x[0]+=1
    else :
      x = x/sumx
    xT = np.transpose(x)
    esp = np.dot(xT, ER)
    var = np.dot(xT, np.dot(SIGMA, x))
    sigma = np.sqrt(var)

    if sigma <= 0.5:
        Axe1.append(sigma)
        Axe2.append(esp)

# Tracé de la frontière efficiente
axesigma = np.linspace(min(Axe1), max(Axe1), 200)
axey1 = [FE(v) for v in axesigma]
axey2 = [FNE(v) for v in axesigma]
axey3 = [cste(v) for v in axesigma]

plt.plot(axesigma, axey1, label="Frontière efficiente supérieure")
plt.plot(axesigma, axey2, label="Frontière efficiente inférieure")
plt.plot(axesigma, axey3, label="Droite moyenne")
plt.scatter(Axe1, Axe2, alpha=0.5, label="Portefeuilles simulés")
plt.legend()
plt.xlabel("Risque (Écart-type)")
plt.ylabel("Espérance de rendement")
plt.title("Frontière efficiente des portefeuilles")
plt.show()

# Simulation de portefeuilles aléatoires
NS = 100000
sigmax = 0.8  # Seuil de risque maximal
espmax = -np.inf
xopt = None

for _ in range(NS):
    x = np.random.dirichlet(np.ones(num_assets))
    xT = np.transpose(x)
    var = np.dot(xT, np.dot(SIGMA, x))
    sigma = np.sqrt(var)

    if sigma < sigmax:
        esp = np.dot(xT, ER)
        if esp > espmax:
            espmax = esp
            xopt = x
            
print("Portefeuille optimal:", xopt)
print("Espérance de rendement:", espmax)

# Stratégie d'investissement

C = [200000]
Cash = []
k = 5 / 100  # Coût de transaction 5%
quant_F = []
Cash_F = []
C_F = [200000]

for j in range(len(BD[0]) - 1):
    quant = [int(C[j] * xopt[i] / ((BD[i].iloc[j, 3] + BD[i].iloc[j, 4]) / 2)) for i in range(num_assets)]
    Cash.append(C[j] - sum([quant[i] * ((BD[i].iloc[j, 3] + BD[i].iloc[j, 4]) / 2) for i in range(num_assets)]))
    C.append(float(Cash[j] + sum([quant[i] * ((BD[i].iloc[j + 1, 3] + BD[i].iloc[j + 1, 4]) / 2) for i in range(num_assets)])))

for j in range(len(BD[0]) - 1):
    quant_F.append([int((C_F[j] * xopt[i] / ((BD[i].iloc[j, 3] + BD[i].iloc[j, 4]) / 2))) for i in range(num_assets)])
    if j == 0:
        CT = 0
    else:
        CT = sum([(k * abs(quant_F[j - 1][i] - quant_F[j][i]) * ((BD[i].iloc[j + 1, 3] + BD[i].iloc[j + 1, 4]) / 2)) for i in range(num_assets)])
    Cash_F.append(C_F[j] - sum([quant_F[j][i] * ((BD[i].iloc[j, 3] + BD[i].iloc[j, 4]) / 2) for i in range(num_assets)]) - CT)
    C_F.append(float(Cash_F[j] + sum([quant_F[j][i] * ((BD[i].iloc[j + 1, 3] + BD[i].iloc[j + 1, 4]) / 2) for i in range(num_assets)])))

print(Cash[-1], C[-1], Cash_F[-1], C_F[-1])

# Graphique des performances
plt.plot([x for x in range(len(C))], C, label="Sans coût de transaction")
plt.plot([x for x in range(len(C_F))], C_F, color="red", label="Avec coût de transaction 5%")
plt.legend()
plt.show()
