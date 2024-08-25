import math

class LinearAlgebra:

    def __init__(self) -> None:
        pass

    def isMultiDArray(self, matrix):
        cols = len(matrix[0])
        for row in matrix:
            if len(row) != cols:
                return False
        return True

    def get_row_col(self, matrix):
        row = len(matrix)
        if self.isMultiDArray(matrix) == True:
            cols = len(matrix[0])
        else:
            cols = None
        return row, cols

    def isSingular(self, matrix):
        if self.isSquare(matrix) == False:
            return False

        # det A = 0
        if self.determinant(matrix) == 0:
            return True
        else:
            return False

    def isSquare(self, matrix):
        row, col = self.get_row_col(matrix)
        if row == col:
            return True
        else:
            return False

    def isDiagonal(self, matrix):
        if self.isSquare(matrix) == False:
            return False

        row, col = self.get_row_col(matrix)
        for i in range(row):
            for j in range(col):
                if i!=j and matrix[i][j] != 0:
                    return False
        return True

    # def isScalar(self, matrix):
    #     # A = kI
    #     pass

    def isIdentity(self, matrix):
        if self.isSquare(matrix) == False:
            return False

        row, col = self.get_row_col(matrix)
        for i in range(row):
            for j in range(col):
                if (i!=j and matrix[i][j] != 0) or (i==j and matrix[i][j] != 1):
                    return False
        return True

    # def isZero(self, matrix):
    #     pass

    def isSymmetric(self, matrix):
        # A = At
        if self.isSquare(matrix) == False:
            return False
        return matrix == self.transpose(matrix)

    def isSkewSymmetric(self, matrix):
        # A = -At
        if self.isSquare(matrix) == False:
            return False
        return -matrix == self.transpose(matrix)

    # def isOrthogonal(self, matrix):
    #     pass

    # def isHermitian(self, matrix):
    #     # A complex square matrix that is equal to its own conjugate transpose.
    #     # A = A*
    #     pass

    # def isSkewHermitian(self, matrix):
        # A = -A*
        # pass

    def isUpperTriangular(self, matrix):
        if self.isSquare(matrix) == False:
            return False
        row,col = self.get_row_col(matrix)
        for i in range(row):
            for j in range(col):
                if i>j and matrix[i][j] != 0:
                    return False
        return True

    def isLowerTriangular(self, matrix):
        if self.isSquare(matrix) == False:
            return False
        row,col = self.get_row_col(matrix)
        for i in range(row):
            for j in range(col):
                if i<j and matrix[i][j] != 0:
                    return False
        return True

    def multiplication(self, matrix1,matrix2):
        r1, c1 = self.get_row_col(matrix1)
        r2, c2 = self.get_row_col(matrix2)
        if(c1!=r2):
            print("number of columns in matrix1 must be equal to number of rows in matrix2")
            return
        else:
            output_matrix = []
            for i in range(r1):
                temp = []
                for j in range(c2):
                    s = 0
                    for k in range(r2):
                        s += matrix1[i][k] * matrix2[k][j]
                    temp.append(s)
                output_matrix.append(temp)
        return output_matrix

    def transpose(self, matrix):
        if(matrix==[]):
            return matrix
        row, col = self.get_row_col(matrix)
        return [[matrix[j][i] for j in range(row)] for i in range(col)]

    def determinant(self, matrix):
        if self.isSquare(matrix) == False:
            raise Exception("Matrix must be square to calculate determinant")
        if(len(matrix) == 1): #base case for 1x1 matrix
            return matrix[0][0]
        # if(len(matrix) == 2): #base case for 2x2 matrix
        #     return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for i in range(len(matrix)):
            submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
            sub_det = self.determinant(submatrix)
            sign = (-1)**i
            det += (sign*matrix[0][i]*sub_det)
        return det

    def lu_decomposition(self, matrix):

        n = len(matrix)
        # initialisations
        l = [[1 if i==j else 0 for j in range(n)] for i in range(n)] # can be made using l = np.eye(n) identity matrix
        u = matrix[:]

        for i in range(n-1):
            diag_el = u[i][i]
            j = i+1
            while(j < n):
                factor = u[j][i] / diag_el
                for k in range(n):
                    u[j][k] = u[j][k] - factor * u[i][k]
                l[j][i] = factor
                j += 1

        return l,u

    def determinant_using_lu_decomposition(self, matrix):
        l, u = self.lu_decomposition(matrix)
        det = 1
        for i in range(len(l)):
            det = det * l[i][i] * u[i][i]
        return det

    def row_echelon_form(self, matrix):
        rows, cols = self.get_row_col(matrix)

        for r in range(rows):
            pivot = r

            # find pivot for curr row
            i = r # assuming the current row contains pivot
            while i<rows-1 and matrix[i][pivot] == 0:
                i += 1
                if(pivot == cols):
                    break

            if(matrix[i][pivot] == 0):
                break

            # if my current row doesn't have non zero at r,r we need to swap rows with ith index
            for j in range(cols):
                temp = matrix[r][j]
                matrix[r][j] = matrix[i][j]
                matrix[i][j] = temp

            # now my current row has a non zero element at the desired position
            # we have to make that pivot as 1, so divide the whole row by that number
            factor = matrix[r][pivot]
            for j in range(cols):
                matrix[r][j] = matrix[r][j] / factor

            # now make all the rows below the current row as 0 for the pivot column
            for r_ in range(r+1, rows):
                factor = matrix[r_][pivot] / matrix[r][pivot]
                for k in range(cols):
                    matrix[r_][k] = matrix[r_][k] - factor * matrix[r][k]

        return matrix

    def rank(self, matrix):
        matrix = self.row_echelon_form(matrix)
        count = 0
        for row in matrix:
            temp = [0.0]*len(row)
            if(row != temp):
                count += 1
        return count

    def cofactor(self, matrix):
        cofactor_matrix = []
        row, col = self.get_row_col(matrix)
        for i in range(row):
            temp = []
            for j in range(col):
                submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
                temp.append((-1)**(i+j)*self.determinant(submatrix))
            cofactor_matrix.append(temp)
        return cofactor_matrix

    def inverse_using_cofactors(self, matrix):
        det = self.determinant(matrix)
        if det == 0:
            return "Matrix is singular and cannot be inverted"

        # Get cofactor matrix
        cofactor_matrix = self.cofactor(matrix)

        # Transpose to get adjugate matrix
        adjugate = self.transpose(cofactor_matrix)

        # Divide adjugate by determinant
        inverse = [[element / det for element in row] for row in adjugate]
        return inverse

    # def inverse_using_gaussian_elimination(self, matrix):
    #     pass

    def flatten(self, matrix):
        vector = []
        row, col = self.get_row_col(matrix)
        for i in range(row):
            vector.extend(matrix[i])
        return vector

    def reshape(self, matrix, new_r, new_c):
        row, col = self.get_row_col(matrix)
        if row*col != new_r*new_c:
            return "This matrix cannot be reshaped into desired shape"
        flattened = self.flatten(matrix)
        new_matrix = []
        for i in range(new_r):
            start_index = i * new_c
            end_index = start_index + new_c
            new_matrix.append(flattened[start_index:end_index])

        return new_matrix

    def trace(self, matrix):
        if self.isSquare(matrix) == False:
            return "Matrix must be square"
        trace = 0
        for i in range(len(matrix)):
            trace += matrix[i][i]
        return trace

    def norm(self, matrix, ord=2):
        row, col = self.get_row_col(matrix)
        if(ord == 1):
            column_sums = [0]*col
            for c in range(col):
                for r in range(row):
                    column_sums[c] += abs(matrix[r][c])
            return max(column_sums)

        elif(ord == 2 or ord == 'fro' or ord == 'spectral'):
            l2_norm = 0
            for r in range(row):
                for c in range(col):
                    l2_norm += (matrix[r][c])**2
            return l2_norm**(1/2)

        elif(ord == -1):
            column_sums = [0]*col
            for c in range(col):
                for r in range(row):
                    column_sums[c] += abs(matrix[r][c])
            return min(column_sums)

        elif(ord == -2):
            return "Work in progress"

        elif(ord == 'inf'):
            row_sums = [0]*row
            for r in range(row):
                for c in range(col):
                    row_sums[r] += abs(matrix[r][c])
            return max(row_sums)

        elif(ord == '-inf'):
            row_sums = [0]*row
            for r in range(row):
                for c in range(col):
                    row_sums[r] += abs(matrix[r][c])
            return min(row_sums)

    def eigen_values(self, matrix):
        """ QR Algorithm for finding eigenvalues of matrix A. """
        iterations = 100
        for _ in range(iterations):
            Q, R = self.qr_decomposition(matrix)
            matrix = self.multiplication(R, Q)  # New matrix for the next iteration

        return [matrix[i][i] for i in range(len(matrix))]

    def householder_reflection(self, A):
        """ Perform QR decomposition using Householder reflections. """
        (m, n) = (len(A), len(A[0]))
        Q = [[float(i == j) for j in range(m)] for i in range(m)]
        R = [row[:] for row in A]

        for k in range(n):
            # Create the vector for the Householder reflection
            x = [R[i][k] for i in range(k, m)]
            norm_x = math.sqrt(sum(x_i**2 for x_i in x))
            r = -math.copysign(norm_x, x[0])
            v = [x_i + r if i == 0 else x_i for i, x_i in enumerate(x)]
            s = math.sqrt(sum(v_i**2 for v_i in v))

            if s != 0:
                u = [v_i / s for v_i in v]
                # Apply the transformation to R and Q
                for j in range(k, n):
                    prod = sum(u[i - k] * R[i][j] for i in range(k, m))
                    for i in range(k, m):
                        R[i][j] -= 2 * u[i - k] * prod
                for i in range(m):
                    prod = sum(u[j - k] * Q[i][j] for j in range(k, m))
                    for j in range(k, m):
                        Q[i][j] -= 2 * u[j - k] * prod

        return [list(map(lambda x: -x if x != 0 else x, q)) for q in Q], R

    def qr_decomposition(self, matrix):
        # Assume get_row_col simply returns the dimensions of the matrix
        row, col = self.get_row_col(matrix)

        Q = [[0]*row for _ in range(col)]  # Adjusted dimensions for column-wise filling
        R = [[0]*col for _ in range(col)]

        for j in range(col):
            # Extract the j-th column from matrix
            jth_vector = [matrix[i][j] for i in range(row)]

            for i in range(j):
                # Calculate the dot product of Q[i] and the j-th column of matrix
                R[i][j] = sum(Q[i][k] * matrix[k][j] for k in range(row))

                # Update jth_vector by subtracting the projection of it onto Q[i]
                jth_vector = [jth_vector[k] - R[i][j] * Q[i][k] for k in range(row)]

            # Calculate the norm of the j-th vector
            norm_jth_vector = (sum(x**2 for x in jth_vector))**0.5

            # Avoid division by zero
            if norm_jth_vector == 0:
                raise ValueError("Matrix has linearly dependent columns or is singular")

            # Normalize the j-th orthogonal vector and store in Q
            for k in range(row):
                Q[j][k] = jth_vector[k] / norm_jth_vector

            # Diagonal elements of R are the norms
            R[j][j] = norm_jth_vector

        # Transpose Q to match expected output
        Q_transposed = [[Q[j][i] for j in range(col)] for i in range(row)]
        return Q_transposed, R

    # def singular_value_decomposition(self, matrix):
    #     pass

    # def cholesky_decomposition(self, matrix):
    #     pass

    # def diagonalisation(self, matrix):
    #     pass

    # def eigen_value_decomposition(self, matrix):
    #     pass
