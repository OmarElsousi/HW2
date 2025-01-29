from NumericalMethods import Probability, GPDF

def main():
    """
    Main function to calculate specific probabilities using the Probability function.
    Prints results in the specified format:
    - P(x<105|N(100,12.5))
    - P(x>=μ+2σ|N(100,3))
    """
    # First calculation: P(x < 105 | N(100, 12.5))
    mean1, stDev1, c1 = 100, 12.5, 105
    prob1 = Probability(GPDF, (mean1, stDev1), c1, GT=False)

    # Second calculation: P(x >= μ + 2σ | N(100, 3))
    mean2, stDev2 = 100, 3
    c2 = mean2 + 2 * stDev2
    prob2 = Probability(GPDF, (mean2, stDev2), c2, GT=True)

    # Print results in the specified format
    print(f"P(x<105|N({mean1},{stDev1}))={prob1:.2f}")
    print(f"P(x>=μ+2σ|N({mean2},{stDev2}))={prob2:.2f}")

if __name__ == "__main__":
    main()
