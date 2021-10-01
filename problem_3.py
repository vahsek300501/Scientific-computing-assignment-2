import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import hilbert
import pdb 

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


def row_equilibrate(A,b) :
    n = A.shape[0]
    for i in range (0,n):
        maxVal=0
        j = 0
        while(j<n):
            maxVal = max(maxVal , abs(A[i][j]))
            j+=1
        j = 0
        while(j<n):
            tmp = float(A[i][j])
            A[i][j] = float(tmp/maxVal)
            j += 1
        b[i] = float(b[i]/maxVal)
    return A,b


A = np.array([[1.,1.,2*pow(10,9)],[2.,-1.,pow(10,9)],[1.,2.,0]])
b = np.array([[1.],[1.],[1.]])


print()
print()
print("Question-3")
print("**********")
print()
print()
print()

print("Part-A")
print(GE_pp(A.copy(),b.copy()))
print()
print()

print("Part-B")
print("Row Equilibrated Matrices")
print(row_equilibrate(A.copy(),b.copy()))
print()
print("Row Equilibrated and solved using Gaussian elimination without pivoting")
A_tmp,b_tmp = row_equilibrate(A.copy(),b.copy())
print(GE(A_tmp,b_tmp))