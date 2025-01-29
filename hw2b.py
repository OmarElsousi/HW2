#region imports
from NumericalMethods import Secant
from math import cos
#endregion

#region function definitions
def fn1(x):
    """
    Function definition for fn1: x - 3cos(x)
    :param x: Input value
    :return: Result of the equation
    """
    return x - 3 * cos(x)

def fn2(x):
    """
    Function definition for fn2: cos(2x) * x**3
    :param x: Input value
    :return: Result of the equation
    """
    return cos(2 * x) * x**3

def main():
    """
    Main function to test Secant method for three cases:
    1. fn1: x - 3cos(x) = 0
    2. fn2: cos(2x) * x^3 = 0 (two configurations)
    :return: None, just prints results
    """
    # Case 1: fn1
    try:
        r1 = Secant(fn1, 1, 2, maxiter=5, xtol=1e-4)
        print("root of fn1 = {root:0.4f}, after {iter} iterations".format(root=r1, iter=5))
    except ValueError as e:
        print(f"fn1 did not converge: {e}")

    # Case 2: fn2, longer iterations
    try:
        r2 = Secant(fn2, 1, 2, maxiter=15, xtol=1e-8)
        print("root of fn2 = {root:0.4f}, after {iter} iterations".format(root=r2, iter=15))
    except ValueError as e:
        print(f"fn2 did not converge: {e}")

    # Case 3: fn2, shorter iterations
    try:
        r3 = Secant(fn2, 1, 2, maxiter=3, xtol=1e-8)
        print("root of fn2 (short iteration) = {root:0.4f}, after {iter} iterations".format(root=r3, iter=3))
    except ValueError as e:
        print(f"fn2 (short iteration) did not converge: {e}")

#endregion

if __name__ == "__main__":
    main()
