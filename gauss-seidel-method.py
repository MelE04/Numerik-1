import numpy as np

def relaxation_iteration(A, b, x, w):
    """
    Ein Schritt der Gauss-Seidel-Relaxation.
    """
    n = A.shape[0]
    y = np.copy(x)
    for i in range(n):
        sigma = sum(A[i, j] * y[j] for j in range(i)) + sum(A[i, j] * x[j] for j in range(i+1, n))
        y[i] = (1 - w) * x[i] + (w / A[i, i]) * (b[i] - sigma)
    return y

def relaxation(A, b, x0, eps, w):
    """
    Gauss-Seidel-Relaxation zur Lösung eines LGS mit Abbruchkriterium.
    """
    x_old = x0
    count = 0
    while True:
        x_new = relaxation_iteration(A, b, x_old, w)
        count += 1
        if np.linalg.norm(x_new - x_old, ord=np.inf) < eps:
            break
        x_old = x_new
    return x_new, count

def optimal_omega_experiment(A, b, x_exact, x0, eps, omega_range):
    """
    Findet experimentell den optimalen Relaxationsparameter und zähle die Iterationen.
    (Weil ich keine Lust hatte, das per Hand zu machen.)
    """
    best_omega = None
    min_iterations = float('inf')
    for w in omega_range:
        _, iterations = relaxation(A, b, x0, eps, w)
        if iterations < min_iterations:
            min_iterations = iterations
            best_omega = w
    return best_omega, min_iterations

def calculate_flops(n, iterations):
    """
    Berechne die Anzahl der FLOPs.
    """
    # Für eine Iteration:
    # - Schleife über alle Variablen (n)
    # - Innerhalb: Sigma-Berechnung kostet etwa 2*n Operationen (Addition und Multiplikation)
    # Pro Iteration: n * (2n) FLOPs
    # Gesamt: iterations * FLOPs pro Iteration
    return iterations * 2 * n * n

# Aufgabe (i)
print("Aufgabe (i):")
A = np.array([
    [1/2, -1/4, 0, 0],
    [-1/4, 1/2, -1/4, 0],
    [0, -1/4, 1/2, -1/4],
    [0, 0, -1/4, 1/2]
])
b = np.array([1, 2, 3, 4])
x_exact = np.array([16, 28, 32, 24])
x0 = np.zeros(4)
eps = 1e-5

omega_range = np.arange(1.0, 2.0, 0.01)
best_omega, iterations = optimal_omega_experiment(A, b, x_exact, x0, eps, omega_range)
x_solution, _ = relaxation(A, b, x0, eps, best_omega)
flops = calculate_flops(len(b), iterations)

print(f"Optimaler Relaxationsparameter: {best_omega:.2f}")
print(f"Anzahl Iterationen: {iterations}")
print(f"Anzahl FLOPs: {flops}")
print(f"Lösungsvektor: {x_solution}")

# Aufgabe (ii)
print("\nAufgabe (ii):")
n = 50
h = 0.01
A = np.eye(n) + h * (np.eye(n, k=1) + np.eye(n, k=-1))  # Diagonale und Nachbarn
b = np.ones(n)
x0 = np.zeros(n)
eps = 1e-6

omega_range = np.arange(1.0, 2.0, 0.01)
best_omega, iterations = optimal_omega_experiment(A, b, None, x0, eps, omega_range)
x_solution, _ = relaxation(A, b, x0, eps, best_omega)
flops = calculate_flops(n, iterations)

print(f"Optimaler Relaxationsparameter: {best_omega:.2f}")
print(f"Anzahl Iterationen: {iterations}")
print(f"Anzahl FLOPs: {flops}")
print(f"Lösungsvektor: {x_solution}")