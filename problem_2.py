import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import hilbert
import pdb

##Gaussian Elimination without pivoting

def forward_elimination(A,b):
	matrix_a_rows = A.shape[0]
	matrix_a_cols = A.shape[1]
	matrix_b_rows = b.shape[0]

	rows = 0
	while(rows<matrix_a_cols-1):
		rows_below = rows+1
		while(rows_below < matrix_a_cols):
			divident = A[rows_below][rows]
			divisor = A[rows][rows]
			multiplication_factor = divident/divisor
			for col_num in range(0,matrix_a_cols):
				A[rows_below][col_num] = A[rows_below][col_num] - (multiplication_factor * (A[rows][col_num]))

			b[rows_below] = b[rows_below] - (multiplication_factor*b[rows])
			rows_below += 1
		rows+=1
	return A,b


def back_substitution(A, b):
    n = A.shape[0]
    x = np.zeros((n,1))
    x[n-1] = b[n-1] / A[n-1, n-1]
    for row in range(n-2, -1, -1):
        sums = b[row]
        for j in range(row+1, n):
            sums = sums - A[row,j] * x[j]
        x[row] = sums / A[row,row]
    return x


def GE(A, b):
    A,b = forward_elimination(A,b)
    return back_substitution(A,b)

def GE_pp(A, b):
    n = np.shape(A)[0]
    l = np.zeros(shape=(n,), dtype=int)
    s = np.zeros(shape=(n,))

    tmp = 0
    while(tmp < n):
        l[tmp] = tmp
        sMax = 0
        tmp2 = 0
        while(tmp2 < n):
            if(abs(A[tmp][tmp2]) > sMax):
                sMax = A[tmp][tmp2]
            tmp2 += 1
        s[tmp] = sMax
        tmp += 1

    k = 0
    while(k < n-1):
        rMax = 0
        i = k
        while(i < n):
            r = abs(A[int(l[i])][k]/s[int(l[i])])
            if(r > rMax):
                rMax = r
                j = i
            i += 1

        lTmp = l[k]
        l[k] = l[j]
        l[j] = lTmp

        i = k+1
        while(i < n):
            a_mult = A[int(l[i])][k]/A[int(l[k])][k]
            A[int(l[i])][k] = a_mult
            j = k+1
            while(j < n):
                A[l[i]][j] = A[int(l[i])][j] - a_mult*A[int(l[k])][j]
                j += 1
            i += 1
        k += 1

    k = 0
    while(k<n-1):
        i = k+1
        while(i < n):
            b[l[i]] = b[int(l[i])] - A[int(l[i])][k]*b[int(l[k])]
            i+=1
        k += 1


    x = np.zeros(shape=(n,1))
    x[n-1] = b[int(l[n-1])]/A[int(l[n-1])][n-1]

    for i in range(n-1,-1,-1):
        sum = b[int(l[i])]
        for j in range(i+1,n):
            sum = sum - A[int(l[i])][j]*x[j]
        x[i] = sum/A[int(l[i])][i]

    return x


class Evaluation:
    @staticmethod
    def get_conditional_number(matrix):
        return np.linalg.norm(matrix)*np.linalg.norm(np.linalg.inv(matrix))

    @staticmethod
    def calculate_error(matrix1, matrix2):
        return np.linalg.norm(np.subtract(matrix1,matrix2))/np.linalg.norm(matrix1)

    @staticmethod
    def calculate_residual(A, x_calculated,x_actual, b_actual):
        return np.linalg.norm(np.subtract(np.dot(A, x_calculated),b_actual))/np.linalg.norm(np.dot(A,x_actual))


def ones_matrix(n):
    ones_mat = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i <= j:
                ones_mat[i][j] = 1
            else:
                ones_mat[i][j] = -1
    return ones_mat



inputs = {}
outputs = {}
def generate_input():
    global inputs
    matrix_list = ['random', 'hilbert', 'ones']
    matrix_size = [10, 20, 30, 40]
    

    np.random.seed(0)

    for size in matrix_size:
        matrix_type_dict = {'random': np.random.rand(size, size), 'hilbert': hilbert(size), 'ones': ones_matrix(size)}
        inputs[size] = matrix_type_dict

def calculate_result():
    global outputs,inputs

    for size, matrices in inputs.items():
        x_star = np.ones((size,1))
        size_result = {}
        for matrix_type,matrix in matrices.items():
            required_matrix_output = {}

            A = matrix.copy()
            b = np.dot(matrix,x_star)
            x_calculated_non_pivot = GE(matrix.copy(),b.copy())
            x_calculated_partial_pivot = GE_pp(matrix.copy(),b.copy())
            x_calculated_linalg_function = np.linalg.solve(matrix.copy(),b.copy())

            required_matrix_output['output_without_pivot'] = x_calculated_non_pivot
            required_matrix_output['output_with_partial_pivot'] = x_calculated_partial_pivot
            required_matrix_output['output_function'] = x_calculated_linalg_function

            required_matrix_output['condition_number'] = Evaluation.get_conditional_number(A)
            required_matrix_output['non_pivot_error'] = Evaluation.calculate_error(x_star,x_calculated_non_pivot)
            required_matrix_output['non_pivot_residual'] = Evaluation.calculate_residual(A,x_calculated_non_pivot,x_star,b)
            required_matrix_output['partial_pivot_error'] = Evaluation.calculate_error(x_star,x_calculated_partial_pivot)
            required_matrix_output['partial_pivot_residual'] = Evaluation.calculate_residual(A,x_calculated_partial_pivot,x_star,b)
            required_matrix_output['linalg_function_error'] = Evaluation.calculate_error(x_star,x_calculated_linalg_function)
            required_matrix_output['linalg_function_residual'] = Evaluation.calculate_residual(A,x_calculated_linalg_function,x_star,b)

            size_result[matrix_type] = required_matrix_output
        outputs[size] = size_result

    # return outputs


generate_input()
calculate_result()


print("n    matrix type    condition_number    no_partial_pivot_error    no_partial_pivot_residual    partial_pivot_error    partial_pivot_residual    inbuilt_function_error    inbuilt_function_residual")
print("**   ***********    ****************    **********************    *************************    *******************    **********************    **********************    *************************")
print()
print()
for val,matrices in outputs.items():
    for mat_type,mat in matrices.items():
        print(str(val)+"    "+str(mat_type)+"    "+str(mat['condition_number'])+"      "+str(mat['non_pivot_error'])+"      "+str(mat['non_pivot_residual'])+"      "+str(mat['partial_pivot_error'])+"      "+str(mat['partial_pivot_residual'])+"      "+str(mat['linalg_function_error'])+"      "+str(mat['linalg_function_residual']))
        print()
        print()