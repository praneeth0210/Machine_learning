import numpy as n


def matrix_calculator():
    print("Matrix Calculator")
    print("Operations:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Transpose")
    print("5. Inverse")
    print("6. Exit")


    while True:
        choice = input("Enter operation number (1-6): ")

        if choice == '6':
            print("Exiting calculator.")
            break

        if choice not in {'1', '2', '3', '4', '5'}:
            print("Invalid operation number. Please enter a number between 1 and 6.")
            continue

        if choice == '4' or choice == '5':
            matrix = n.array(eval(input("Enter matrix as a list of lists: ")))
        else:
            matrix1 = n.array(eval(input("Enter first matrix as a list of lists: ")))
            matrix2 = n.array(eval(input("Enter second matrix as a list of lists: ")))

        if choice == '1':
            result = n.add(matrix1, matrix2)
            print("Result:")
            print(result)
        elif choice == '2':
            result = n.subtract(matrix1, matrix2)
            print("Result:")
            print(result)
        elif choice == '3':
            result = n.matmul(matrix1, matrix2)
            print("Result:")
            print(result)
        elif choice == '4':
            result = n.transpose(matrix)
            print("Result:")

            print(result)
        elif choice == '5':
            try:
                inverse = n.linalg.inv(matrix)
                print("Inverse:")
                print(inverse)
            except n.linalg.LinAlgError:
                print("The matrix is singular and does not have an inverse.")


matrix_calculator()
