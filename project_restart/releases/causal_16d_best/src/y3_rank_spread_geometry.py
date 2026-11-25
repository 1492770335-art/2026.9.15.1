"""Incidence matrix and regularized potential map for the within-tick stock rank graph."""
import numpy as np

GAPS=(1,3,5,8,12,17)
LEFT,RIGHT,LEVEL=np.array([(j,j+gap,k) for k,gap in enumerate(GAPS) for j in range(50-gap)]).T
INC=np.eye(50)[RIGHT]-np.eye(50)[LEFT]
LAPLACIAN=INC.T@INC
MEAN=INC/LAPLACIAN.diagonal().mean()
PROJECT=INC@np.linalg.inv(LAPLACIAN+1.5*np.eye(50))
