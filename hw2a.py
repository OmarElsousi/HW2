from NumericalMethods import Probability, GPDF

def main():
    """
    Main function to calculate specific probabilities using the Probability function.
    Prints results in the specified format:
    - P(x<1.00|N(0,1))=
    - P(x>181.00|N(175,3))=
    """
    # First calculation: P(x < 1 | N(0,1))
    mean1, stDev1, c1 = 0, 1, 1
    prob1 = Probability(GPDF, (mean1, stDev1), c1, GT=False)

    # Second calculation: P(x >= 181 | N(175, 3))
    mean2, stDev2 = 175, 3
    c2 = mean2 + 2 * stDev2  # 175 + 2(3) = 181
    prob2 = Probability(GPDF, (mean2, stDev2), c2, GT=True)

    # Print results in the required format
    print(f"P(x<{c1:.2f}|N({mean1:.0f},{stDev1:.0f}))={prob1:.2f}")
    print(f"P(x>{c2:.2f}|N({mean2:.0f},{stDev2:.0f}))={prob2:.2f}")

if __name__ == "__main__":
    main()
