#region imports
import Gauss_Elim as GE  # Module with matrix manipulation functions
from math import sqrt, pi, exp
#endregion

#region function definitions
def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability that x > c or x < c using Simpson's rule.
    :param PDF: the probability density function to integrate
    :param args: (mean, standard deviation)
    :param c: threshold value
    :param GT: True for x > c, False for x < c
    :return: probability value
    """
    mu, sig = args
    if GT:
        lhl, rhl = c, mu + 5 * sig
    else:
        lhl, rhl = mu - 5 * sig, c

    # Adjust Simpson args to include integration bounds
    p = Simpson(PDF, (mu, sig, lhl, rhl))
    return p

def GPDF(args):
    """
    Gaussian probability density function.
    :param args: (x, mean, standard deviation)
    :return: value of GPDF at x
    """
    x, mu, sig = args
    fx = (1 / (sig * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)
    return fx

def Simpson(fn, args, N=100):
    """
    Numerical integration using Simpson's 1/3 rule.
    :param fn: function to integrate
    :param args: (mean, stDev, lhl, rhl) - includes integration bounds
    :param N: number of intervals (default: 100)
    :return: approximate integral
    """
    mu, sig, lhl, rhl = args
    if N % 2 != 0:
        N += 1  # Ensure N is even
    h = (rhl - lhl) / N  # Step size
    total = 0

    for i in range(N + 1):
        x = lhl + i * h
        if i == 0 or i == N:
            total += fn((x, mu, sig))  # f(x) for endpoints
        elif i % 2 == 0:
            total += 2 * fn((x, mu, sig))  # 2*f(x) for even indices
        else:
            total += 4 * fn((x, mu, sig))  # 4*f(x) for odd indices

    return (h / 3) * total

def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of fcn(x) in the neighborhood of x0 and x1.

    :param fcn: The function for which we want to find the root.
    :param x0: Initial guess for the root (first value).
    :param x1: Second guess for the root (second value).
    :param maxiter: Maximum number of iterations to perform.
    :param xtol: Tolerance for convergence; exit if |xnewest - xprevious| < xtol.
    :return: The final estimate of the root (most recent new x value).
    """
    for iteration in range(maxiter):
        # Evaluate the function at the current points
        fx0 = fcn(x0)
        fx1 = fcn(x1)

        # Check for division by zero
        if abs(fx1 - fx0) < 1e-12:
            raise ValueError("Division by zero in the Secant method. Function values at x0 and x1 are too close.")

        # Compute the next value using the Secant formula
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Check for convergence
        if abs(x_new - x1) < xtol:
            return x_new

        # Update x0 and x1 for the next iteration
        x0, x1 = x1, x_new

    # If max iterations are reached without convergence, raise an exception
    raise ValueError("Secant method did not converge within the maximum number of iterations.")

def GaussSeidel(Aaug, x, Niter=15):
    """
    Solve a system of equations using the Gauss-Seidel method.
    :param Aaug: Augmented matrix [A|b] of the system
    :param x: Initial guess for solution
    :param Niter: Number of iterations
    :return: Solution vector x
    """
    n = len(Aaug)
    for _ in range(Niter):
        for i in range(n):
            sum_ax = sum(Aaug[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (Aaug[i][-1] - sum_ax) / Aaug[i][i]
    return x

def main():
    """
    Test numerical methods locally.
    :return: None
    """
    # Test GPDF
    print("Testing GPDF:")
    fx = GPDF((0, 0, 1))  # Standard normal distribution at x=0
    print(f"GPDF(0,0,1) = {fx:.5f}")  # Expected: ~0.39894

    # Test Simpson's rule
    print("\nTesting Simpson's Rule:")
    p = Simpson(GPDF, (0, 1, -5, 0))  # Should return ~0.5 for N=100
    print(f"Integral of GPDF over [-5,0] = {p:.5f}")

    # Test Probability function
    print("\nTesting Probability Function:")
    p1 = Probability(GPDF, (0, 1), 0, True)  # P(x > 0 | N(0,1))
    print(f"P(x>0|N(0,1)) = {p1:.5f}")  # Expected: ~0.5

    # Test Secant method
    print("\nTesting Secant Method:")
    root = Secant(lambda x: x**2 - 4, 1, 3, maxiter=100, xtol=1e-6)
    print(f"Root of x^2 - 4 is {root:.5f}")

    # Test Gauss-Seidel method
    print("\nTesting Gauss-Seidel Method:")
    Aaug = [
        [4, -1, 0, 3],
        [-1, 4, -1, 2],
        [0, -1, 4, 1]
    ]  # Solves system Ax=b for x
    x_guess = [0, 0, 0]
    x_solution = GaussSeidel(Aaug, x_guess)
    print(f"Solution vector x = {x_solution}")

#endregion

#region function calls
if __name__ == '__main__':
    main()
#endregion
