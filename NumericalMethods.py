# region imports
import Gauss_Elim as GE  # Module with matrix manipulation functions
from math import sqrt, pi, exp


# endregion

# region function definitions
def Probability(PDF, args, c, GT=True):
    """
    Calculate the probability that x > c or x < c using Simpson's rule.

    This function integrates the given probability density function (PDF) over a
    specified interval using Simpson's rule. The integration bounds depend on whether
    we want the probability of x being greater than or less than the cutoff value c.

    :param PDF: The probability density function to integrate.
    :param args: A tuple (mean, standard deviation) for the Gaussian PDF.
    :param c: The threshold value for integration.
    :param GT: If True, compute the probability for x > c; if False, for x < c.
    :return: The computed probability value.
    """
    mu, sig = args
    # Define integration limits based on GT flag
    if GT:
        lhl, rhl = c, mu + 5 * sig  # from c to a point far right of the mean
    else:
        lhl, rhl = mu - 5 * sig, c  # from far left of the mean to c

    # Use Simpson's rule to approximate the integral
    p = Simpson(PDF, (mu, sig, lhl, rhl))
    return p


def GPDF(args):
    """
    Gaussian probability density function.

    Computes the value of the Gaussian (normal) probability density function at x.

    :param args: A tuple (x, mean, standard deviation).
    :return: The value of the Gaussian PDF at x.
    """
    x, mu, sig = args
    fx = (1 / (sig * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)
    return fx


def Simpson(fn, args, N=100):
    """
    Approximate the integral of a function using Simpson's 1/3 rule.

    The function integrates fn from a lower bound (lhl) to an upper bound (rhl)
    using N intervals (where N is made even if necessary).

    :param fn: The function to integrate.
    :param args: A tuple (mean, standard deviation, lhl, rhl) that provides the
                 additional parameters and integration bounds.
    :param N: The number of subintervals to use (default is 100).
    :return: The approximate value of the integral.
    """
    mu, sig, lhl, rhl = args
    if N % 2 != 0:
        N += 1  # Ensure N is even for Simpson's rule
    h = (rhl - lhl) / N  # Compute the width of each subinterval
    total = 0

    # Apply Simpson's 1/3 rule
    for i in range(N + 1):
        x = lhl + i * h
        fx = fn((x, mu, sig))  # Evaluate the function at x
        if i == 0 or i == N:
            total += fx  # Endpoints are weighted 1
        elif i % 2 == 0:
            total += 2 * fx  # Even-indexed interior points are weighted 2
        else:
            total += 4 * fx  # Odd-indexed points are weighted 4

    return (h / 3) * total


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Use the Secant Method to find the root of a function near x0 and x1.

    The Secant method is an iterative technique for finding the root of a function.

    :param fcn: The function for which the root is to be found.
    :param x0: The first initial guess for the root.
    :param x1: The second initial guess for the root.
    :param maxiter: Maximum number of iterations allowed.
    :param xtol: Tolerance for convergence; the process stops if the difference between
                 successive estimates is less than xtol.
    :return: The estimated root.
    :raises ValueError: If division by zero occurs or the method fails to converge.
    """
    for iteration in range(maxiter):
        fx0 = fcn(x0)
        fx1 = fcn(x1)

        # Check for potential division by zero
        if abs(fx1 - fx0) < 1e-12:
            raise ValueError("Division by zero in the Secant method. Function values at x0 and x1 are too close.")

        # Compute the next approximation using the Secant formula
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        # Check if the current estimate is within the specified tolerance
        if abs(x_new - x1) < xtol:
            return x_new

        # Update guesses for the next iteration
        x0, x1 = x1, x_new

    # If convergence is not reached within maxiter, raise an error
    raise ValueError("Secant method did not converge within the maximum number of iterations.")


def GaussSeidel(Aaug, x, Niter=15):
    """
    Solve a system of linear equations using the Gauss-Seidel iterative method.

    This function solves the system represented by the augmented matrix Aaug using the
    provided initial guess x. It performs a specified number of iterations (Niter).

    :param Aaug: The augmented matrix [A|b] representing the system of equations.
    :param x: Initial guess vector for the solution.
    :param Niter: Number of iterations to perform (default is 15).
    :return: The solution vector x after Niter iterations.
    """
    n = len(Aaug)
    for _ in range(Niter):
        for i in range(n):
            # Compute the sum of A[i][j] * x[j] for all j != i
            sum_ax = sum(Aaug[i][j] * x[j] for j in range(n) if j != i)
            # Update the i-th variable using the rearranged formula
            x[i] = (Aaug[i][-1] - sum_ax) / Aaug[i][i]
    return x


def main():
    """
    Test the numerical methods implemented in this module.

    This function runs several tests:
      - Evaluates the Gaussian PDF (GPDF) at a specific point.
      - Uses Simpson's rule to integrate the Gaussian PDF.
      - Computes a probability using the Probability function.
      - Finds a root using the Secant method.
      - Solves a system of equations using the Gauss-Seidel method.
    """
    # Test the Gaussian PDF (GPDF)
    print("Testing GPDF:")
    fx = GPDF((0, 0, 1))  # Evaluate the standard normal PDF at x=0
    print(f"GPDF(0, 0, 1) = {fx:.5f}")  # Expected value: ~0.39894

    # Test Simpson's rule for numerical integration
    print("\nTesting Simpson's Rule:")
    p = Simpson(GPDF, (0, 1, -5, 0))  # Integrate standard normal PDF from -5 to 0
    print(f"Integral of GPDF over [-5,0] = {p:.5f}")

    # Test the Probability function for calculating tail probabilities
    print("\nTesting Probability Function:")
    p1 = Probability(GPDF, (0, 1), 0, GT=True)  # Compute P(x > 0) for N(0,1)
    print(f"P(x>0|N(0,1)) = {p1:.5f}")  # Expected: ~0.5

    # Test the Secant method for finding roots
    print("\nTesting Secant Method:")
    root = Secant(lambda x: x ** 2 - 4, 1, 3, maxiter=100, xtol=1e-6)
    print(f"Root of x^2 - 4 is {root:.5f}")

    # Test the Gauss-Seidel method for solving systems of equations
    print("\nTesting Gauss-Seidel Method:")
    Aaug = [
        [4, -1, 0, 3],
        [-1, 4, -1, 2],
        [0, -1, 4, 1]
    ]  # Represents the system Ax = b
    x_guess = [0, 0, 0]  # Initial guess for the solution vector
    x_solution = GaussSeidel(Aaug, x_guess)
    print(f"Solution vector x = {x_solution}")


# endregion

# region function calls
if __name__ == '__main__':
    main()
# endregion
