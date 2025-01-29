from NumericalMethods import GaussSeidel

def make_diag_dom(Aaug):
    """
    Rearrange rows of the augmented matrix Aaug to make it diagonally dominant.
    :param Aaug: Augmented matrix [A|b]
    :return: Augmented matrix rearranged to be diagonally dominant, if possible.
    """
    n = len(Aaug)
    for i in range(n):
        max_row = i
        for j in range(i, n):
            if abs(Aaug[j][i]) > abs(Aaug[max_row][i]):
                max_row = j
        Aaug[i], Aaug[max_row] = Aaug[max_row], Aaug[i]
    return Aaug

def GaussSeidel(Aaug, x, Niter=15):
    """
    Solve a system of equations using the Gauss-Seidel method.
    :param Aaug: Augmented matrix [A|b] of the system
    :param x: Initial guess for solution
    :param Niter: Number of iterations
    :return: Solution vector x
    """
    n = len(Aaug)
    Aaug = make_diag_dom(Aaug)  # Ensure the matrix is diagonally dominant
    for _ in range(Niter):
        for i in range(n):
            sum_ax = sum(Aaug[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (Aaug[i][-1] - sum_ax) / Aaug[i][i]
    return x

def main():
    """
    Solve two sets of linear equations using the GaussSeidel function.
    :return: None
    """
    # First system of equations
    Aaug1 = [
        [3, 1, -1, 2],
        [1, 4, 1, 12],
        [2, 1, 2, 10]
    ]
    x_guess1 = [0, 0, 0]  # Initial guess
    solution1 = GaussSeidel(Aaug1, x_guess1, Niter=15)
    print("Solution to the first system of equations:")
    print(solution1)

    # Second system of equations
    Aaug2 = [
        [1, -10, 2, 4, 2],
        [3, 1, 4, 1, 12],
        [9, 2, 3, 4, 21],
        [-1, 2, 7, 3, 37]
    ]
    x_guess2 = [0, 0, 0, 0]  # Initial guess
    solution2 = GaussSeidel(Aaug2, x_guess2, Niter=15)
    print("\nSolution to the second system of equations:")
    print(solution2)

if __name__ == "__main__":
    main()
