import numpy as np

x = np.array([-1.6])
A3 = [0,0]
for j in range(0,100):
    
    x_new = x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j]))/(np.sin(3 * x[j]) + 3 * x[j] * np.cos(3*x[j]) - np.exp(x[j]))

    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    x = np.append(x, x_new)
    A3[0] = j + 1
    if (abs(fc) < 1e-6):
        break
A1 = x
A1 

xr = -0.4; xl = -0.7
A2 = []
for j in range(0, 100):
    xc = (xr + xl)/2
    fc = xc * np.sin(3 * xc) - np.exp(xc)
    if fc > 0:
        xl = xc   
    else:
        xr = xc
    A2.append(xc)
    A3[1] = j + 1
    if (abs(fc) < 1e-6):       
        break
A3

A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])
A4 = np.add(A,B)
A5 = 3*x-4*y
A6 = np.dot(A,x)
A7 = np.dot(B,x-y)
A8 = np.dot(D,x)
A9 = np.add(np.dot(D,y),z)
A10 = np.dot(A,B)
A11 = np.dot(B,C)
A12 = np.dot(C,D)
