# region imports
from NumericalMethods import Secant  # Import the Secant method from the NumericalMethods module
from math import cos  # Import the cosine function from the math module


# endregion

# region function definitions
def fn1(x):
    """
    Compute the function: f(x) = x - 3cos(x)

    :param x: Input value
    :return: Result of x - 3cos(x)
    """
    return x - 3 * cos(x)


def fn2(x):
    """
    Compute the function: f(x) = cos(2x) * x^3

    :param x: Input value
    :return: Result of cos(2x) * x^3
    """
    return cos(2 * x) * x ** 3


def main():
    """
    Main function to test the Secant method on two functions:

    Case 1: Finds the root of fn1 (i.e., solves x - 3cos(x) = 0)
    Case 2: Finds the root of fn2 (i.e., solves cos(2x) * x^3 = 0) with two different iteration limits.

    Results are printed with the estimated root and the number of iterations used.
    """
    # Case 1: Solve fn1 using the Secant method with a maximum of 5 iterations.
    try:
        r1 = Secant(fn1, 1, 2, maxiter=5, xtol=1e-4)
        print("root of fn1 = {root:0.4f}, after {iter} iterations".format(root=r1, iter=5))
    except ValueError as e:
        print(f"fn1 did not converge: {e}")

    # Case 2: Solve fn2 with a higher maximum iteration count (15 iterations) for better convergence.
    try:
        r2 = Secant(fn2, 1, 2, maxiter=15, xtol=1e-8)
        print("root of fn2 = {root:0.4f}, after {iter} iterations".format(root=r2, iter=15))
    except ValueError as e:
        print(f"fn2 did not converge: {e}")

    # Case 3: Solve fn2 with a lower maximum iteration count (3 iterations) to test short iteration behavior.
    try:
        r3 = Secant(fn2, 1, 2, maxiter=3, xtol=1e-8)
        print("root of fn2 (short iteration) = {root:0.4f}, after {iter} iterations".format(root=r3, iter=3))
    except ValueError as e:
        print(f"fn2 (short iteration) did not converge: {e}")


# endregion

if __name__ == "__main__":
    main()
