import numpy as np
import matplotlib.pyplot as plt 

def create_img(r):
    mat = np.ones((4*r,4*r))
    
    ###creating the triangles using D-4 distance

    #left triangle, center at (r,2r) and distance = r
    for i in range(4*r):
        for j in range(0,r):
            if abs(i-2*r)+abs(j-r)<r:             #add one for adjustment
                mat[i,j]=0

    #left triangle, center at (r,2r) and distance = r
    for i in range(4*r):
        for j in range(3*r,4*r):
            if abs(i-2*r)+abs(j-3*r)<r:             #add one for adjustment
                mat[i,j]=0

    ###creating the center square

    #complete square with side length 2r
    for i in range(r,3*r):
        for j in range(r,3*r):
            mat[i,j]=0
    
    #carving out a smaller square of side length r
    for i in range(3*r//2, 3*r//2+r):
        for j in range(3*r//2, 3*r//2+r):
            mat[i,j]=1

    ##creating top and bottom hemi-circles using Euclidean distance

    #top hemi-cirlce using center (2r,r) and distance = r
    for i in range(0,r):                     #range of length r to create hemi-circle
        for j in range(0,4*r):
            if (i-r)**2+(j-2*r)**2<=r**2:
                mat[i,j]=0

    #bottom hemi-circle using center (2r,3r) and distance = r
    for i in range(3*r,4*r):                     #range of length r to create hemi-circle
        for j in range(0,4*r):
            if (i-3*r)**2+(j-2*r)**2<=r**2:
                mat[i,j]=0

    # Add a border of length r
    final_mat = np.ones((6*r, 6*r))
    final_mat[r:5*r, r:5*r] = mat

    plt.imshow(final_mat, cmap='gray')
    plt.show()
    # return final_mat.astype(float)

# Generate and show the plot for r = 100
create_img(1000)


