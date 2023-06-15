import numpy as np

def dotProduct(a, b):
    # Perform the dot product using NumPy
    dot_product = np.dot(a, b)

    return dot_product

def main():
    # Test the dot product function
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0])

    result = dotProduct(a, b)
    print("Dot Product:", result)

if __name__ == "__main__":
    main()
