import numpy as np

# LU-Zerlegung für tridiagonale Matrizen

def tridiag_lu(A):
    for i in range(len(A[0])-1): # zum Ermitteln der Ordnung n
        if A[1][i] == 0:
            return "Error: Can divide by 0" # Fehlermeldung, wenn das Pivotelement 0 wird
        A[0][i]   = A[0][i]/A[1][i] # schreibt L in die 1. Zeile
        A[1][i+1] = A[1][i+1] - A[0][i] * (A[2][i+1]) # berechnet U
    return A

# Vorwaerts - und Rueckwaertselimination einer LU - Zerlegung

def tridiag_vwrw(Z,b):
    # Vorwaertseinsetzen
    y = [b[0]] # Hilfsvektor y haelt die Ergebnisse vom Vorwaertseinsetzen
    for i in range(1,len(b)): # zum Ermitteln der Ordnung n
        y.append(b[i]-Z[0][i-1] * y[i-1])
    # Rueckwaertseinsetzen
    x = [None] * (len(b) - 1) + [y[i]/Z[1][i]] # Initialisiere x mit Platzhaltern und füge das erste Ergebnis vom Rueckwaertseinsetzen hinten ein
    for i in range(len(b)-2,-1,-1):
        x[i] = (y[i]-x[i+1]*Z[2][i+1])/Z[1][i]
    return x


# 1. AUFGABE

A = [[ 4, 5, 6, 0],
     [10,20,30,40],
     [ 0, 1, 2, 3]]

b = [12,50,112,178]

print(tridiag_vwrw(tridiag_lu(A),b))


# 2. AUFGABE

def makeA(n):
    return [[-1]*(n-1) + [0], [2]*n, [0] + [-1]*(n-1)]

def makeb(n):
    return [0.01] + [0]*(n-2) + [0.01]

# Testen der Lösung mit den exakten Werten
# n_values = [10, 100, 1000]
n_values = [10] # zum einfacheren Testen
exact_solution = np.array([0.01] * max(n_values))  # Exakte Lösung für alle n-Werte

for n in n_values:
    A = makeA(n)
    b = makeb(n)
    
    # Berechnung der Näherungslösung
    approx_solution = tridiag_vwrw(tridiag_lu(A), b)
    approx_solution = np.array(approx_solution)
    
    # Relativer Fehler in der unendlichen Norm
    relative_error = np.linalg.norm(approx_solution - exact_solution[:n], np.inf) / np.linalg.norm(exact_solution[:n], np.inf)
    print(f"Für n={n}: Näherungslösung={approx_solution}, Relativer Fehler={relative_error:.8f}")
