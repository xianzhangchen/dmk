"""
This code is the first test python code, copy from http://www.guanjihuan.com
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import pyx

def matrix_00(width=10):  # default value is 10
    h00 = np.zeros((width,width))
    for width0 in range(width-1):
        h00[width0,width0+1] = 1
        h00[width0+1,width0] = 1
    return h00 

def matrix_01(width=10): # default value is 10
    h01 = np.identity(width)
    return h01

"""
np.identity(n,dtype=None)
paramters: n, int型表示输出的矩阵的行数和列数都是n
dtype: 表示输出的类型，默认是float
return: 返回的是nXn的主对角线为1，其余地方为0的数组
"""

def main():
    start_time = time.time()
    h00 = matrix_00(width=5)
    h01 = matrix_01(width=5)
    fermi_energy_array = np.arange(-4,4,.01)
    plot_conductance_energy(fermi_energy_array,h00,h01)
    end_time = time.time()
    print('run time:',end_time-start_time)

def plot_conductance_energy(fermi_energy_array,h00,h01):
    """
    plot the conductance as a function of fermi energy energy
    """
    dim = fermi_energy_array.shape[0]
    cond = np.zeros(dim)
    i0 = 0
    for fermi_energy0 in fermi_energy_array:
        cond0 = np.real(conductance(fermi_energy0+1e-12j,h00,h01))
        cond[i0] = cond0
        i0 += 1
    plt.plot(fermi_energy_array,cond,'-k')
    plt.show()

def transfer_matrix(fermi_energy,h00,h01,dim): 
    """
    transfer matrix T, dim is the dimensional of transmission matrix h00 and h01
    """
    transfer = np.zeros((2*dim,2*dim),dtype=complex)
    transfer[0:dim,0:dim] = np.dot(np.linalg.inv(h01), fermi_energy*np.identity(dim)-h00) # np.dot equal to np.matul()
    transfer[0:dim,dim:2*dim] = np.dot(-1*np.linalg.inv(h01), h01.transpose().conj())
    transfer[dim:2*dim,0:dim] = np.identity(dim)
    transfer[dim:2*dim,dim:2*dim] = 0 # a:b is a <= x < b, left close and right open
    return transfer
