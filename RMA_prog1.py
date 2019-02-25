# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:04:38 2018
----------------------------------------
----- ALGORITMO RANGE-MIGRATION ------
                v1.0
----------------------------------------
@author: LUIS
"""

import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import time
import sarPrm as sp # Created Library to define sistem parameters
import drawFigures as dF # Created Library to draw FFT functions
from angles import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

#%%
#########################################################
#     GENERACIÓN DE LOS DATOS (SIMULACIÓN DIRECTA)      #
#########################################################

# N° de puntos: 1

#--------------PARAMETERS DEFINITION---------------------
c,fc,BW,Nf,Ls,Np,Ro = sp.get_parameters() # Parametros
Rt_x, Rt_y= sp.get_scalar_target() # Coordendas del target(m)
Ru_m, dx_a_m = sp.verify_conditions() # Verificar rango y paso del riel maximo
rr_r, rr_x= sp.get_resolutions(Ru_m) # Resoluciones en rango y azimuth
#Ru_m, dx_a_m = sp.verify_conditions() # Verificar rango y paso del riel maximo

df=BW/(Nf-1) # Paso de frecuencia(GHz)
dp=Ls/(Np-1) # Paso del riel(m)
fi=fc-BW/2 # Frecuencia inferior(GHz)
fs=fc+BW/2 # Frecuencia superior(GHz)

print("--------INFORMACIÓN IMPORTANTE--------------")
print("Range_resolution: ", rr_r)
print("Cross_range_resolution: ", rr_x)
print("--------------------------------------------")
print("Rango máximo (m): ", Ru_m)
print("Paso del riel máximo(mm): ", dx_a_m*1000)
print("______¿Se cumplen las condiciones?___________")
print("Rango < maximo?: ", Rt_x<=Ru_m)
print("Paso del riel < máximo?: ", dp<=dx_a_m)
print("Condicion de interpolacion?:", dp>(c/(4*fi))) # Para evitar que salgan valores imaginarios al cambiar los ejes de k-->kx

#------------ ARRAY, VECTOR DEFINITIONS-----------------
class array_data:
    def __init__(self, fi_, df_, Nf_, Ls_, paso_r_, Np_, rt_x_, rt_y_):
        self.fi=fi_
        self.df=df_
        self.Nf=Nf_
        self.Ls=Ls_
        self.paso_r=paso_r_
        self.Np=Np_
        self.rt_x=rt_x_
        self.rt_y=rt_y_

    # Frequency vector (SFCW)
    def getFreq(self):
        return np.array([self.fi+m*self.df for m in range(self.Nf)])
    # Distance vector of riel_k position
    def getRiel_pos(self):
        return np.array([(-self.Ls/2 + k*self.paso_r) for k in range(self.Np)])
    # Distance vector between target and riel_k position
    def getDist(self):
        return np.array([(self.rt_x**2+(self.rt_y - (-self.Ls/2 + k*self.paso_r))**2)**0.5 for k in range(self.Np)])

data=array_data(fi, df, Nf, Ls, dp, Np, Rt_x, Rt_y)
Lista_f=data.getFreq() # Vector de frecuencias
Lista_pos=data.getRiel_pos() # Vector de posiciones del riel
Ri=data.getDist() # Vector de distancias del riel al target

#--------------------PHASE HISTORICAL---------------------------
Sr_f=np.array([np.exp(-1j*4*np.pi*fi*ri/c) for ri in Ri for fi in Lista_f]) # Create a vector with value for each fi y ri
Sr_f=np.reshape(Sr_f,(Np,Nf)) # Reshape the last vector Sr_f

# GRAFICAS
plt.close('all') # Cerrar todas las figuras previas

cmap="plasma"
fig, ax = plt.subplots(1,2)
# Magnitud del historico de fase
im1=ax[0].imshow(20*np.log10(abs(Sr_f)),cmap,origin='lower', extent=[fi,fs,-Ls/2,Ls/2])#vmin=-600,vmax=-100)
ax[0].set(xlabel='Frecuencia(GHz)',ylabel='Posición del riel(m)', title='Histórico de fase (Magnitud)\n')
divider1 = make_axes_locatable(ax[0])
cax = divider1.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
plt.colorbar(im1,cax=cax,label='dBu',extend='both')
# Fase del historico de fase
im2=ax[1].imshow(np.angle(Sr_f),cmap,origin='lower', extent=[fi,fs,-Ls/2,Ls/2])#vmin=-600,vmax=-100)
ax[1].set(xlabel='Frecuencia(GHz)',ylabel='Posición del riel(m)', title='Histórico de fase (Fase)\n')
divider2 = make_axes_locatable(ax[1])
cax = divider2.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
plt.colorbar(im2,cax=cax,label='rad',extend='both')
plt.subplots_adjust(wspace=0.8) # Da un espacio vertical entre las subgraficas

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

######################################################### 
#        PROCESAMIENTO Y GENERACION DE LA IMAGEN        #
#########################################################
#
start_time = time.clock()

#------------------PARAMETROS----------------------------
# Definiendo el area de la imagen 
Lx=40 #(m)
dx=0.004#0.008 #(m)
Nx=Lx/dx
Ly=20 #(m)
dy=0.004#0.008 #(m)
Ny=Ly/dy

#---------------ETAPA PREVIA: IFFT respecto a 'w'---------------------
#---------------------------------------------------------------------
#S0=np.fft.ifft(Sr_f,axis=1)
#cmap="plasma"
#fig, ax = plt.subplots()
#im=ax.imshow(20*np.log10(abs(S0)),cmap,origin='lower',extent=[0,rr_r*Nf,-Ls/2,Ls/2],aspect='auto')
#x3=Lista_pos
#y3=Sr_f.T[0]
#a,b=dF.plotFFTc_rk(x3,y3,x_unit='cm')
# Dibujar el range profile 
#for i in range(len(Sr_f)):
dF.rangeProfile(BW,Sr_f[50])




dF.crangeProfile(rr_x,Sr_f.T[10])


#%%---------------PRIMERA ETAPA: FFT respecto a 'u'---------------------
#                  s(u,w) --> s(ku,w)
#---------------------------------------------------------------------
#S1_a=np.array([np.fft.fft(Sr_f[:,j]) for j in range(len(Lista_f))]).T
Sr_f *= np.hanning(Nf)
S1_aux=np.fft.fft(Sr_f,axis=0) # axis 0 representa las filas(axis 1 a las columnas), lo cual quiere decir q se sumaran todas las filas manteniendo constante las columnas
S1=np.fft.fftshift(S1_aux,axes=(0))
#fig, ax = plt.subplots()
#im=ax.imshow(20*np.log10(abs(np.fft.ifft2(S1))),cmap,origin='lower',extent=[0,rr_r*Nf,-Ls/2,Ls/2],aspect='auto')
# !!Verificar si estan correctamente alineados 

#%%---------------SEGUNDA ETAPA: Matched Filter---------------------
#                  s(ku,w) --> F(kxmn,kymn)
#-------------------------------------------------------------------
k=2*np.pi*Lista_f/c # Numeros de onda dependientes de la frecuencia 
ku=np.linspace(-np.pi/dp,np.pi/dp,Np) # Numeros de onda resultado de la FFT ¿? x q esa expresion(2k_min>ku_max, para realizar el filtro)

# Cambio de variables
kkk, kuu= np.meshgrid(k,ku)
kx=np.sqrt(4*kkk**2-kuu**2) # Genera una matriz, para cada valor de ky [0,....len(ku)-1]
ky=kuu

# Matched Filter / F(k,ku)=s(ku,w)*So
Xc1=Lx/2#np.sqrt((Lx/2)**2 + (Ly/2)**2)
Xc2=0#Ly/2

phi_m= Xc1*np.sqrt(4*kkk**2-kuu**2)#-Xc1*kkk
F1=S1*np.exp(1j*phi_m) # array
F1_m=F1.reshape(1,Np*Nf)[0] # matrix

# Grafica en funcion de ku vs k, despues del fft 
dF.plotImage(S1, x_min=k.min(), x_max=k.max(), y_min=ku.min(), y_max=ku.max(),xlabel_name='k(1/m)', ylabel_name='ku(1/m)', title_name='Gráfica después del 1D-FFT - [xo,yo]=[%i,%i]\n(with fftshift)'%(Rt_x, Rt_y))
# Grafica despues del Matched Filter 
dF.plotImage(F1, x_min=k.min(), x_max=k.max(), y_min=ku.min(), y_max=ku.max(),xlabel_name='k(1/m)', ylabel_name='ku(1/m)', title_name='Gráfica después del Matched Filter - [xo,yo]=[%i,%i]\n(with fftshift)'%(Rt_x, Rt_y))

#%%---------------TERCERA ETAPA: Stolt Interpolation------------------
#                  F(kxmn,kymn) --> F(kx,ky)
#---------------------------------------------------------------------

kr_x=np.linspace(-np.pi/dx,np.pi/dx,int(Nx)) # coordenada kx #np.linspace(0,700,int(Nx))
kr_y=np.linspace(-np.pi/dy,np.pi/dy,int(Ny)) # coordenada ky np.linspace(-600,600, int(Nx))
kr_r=np.array([(i,j) for j in kr_y for i in kr_x]) # coordenada final 

# Interpolacion con griddata
kr_xx, kr_yy = np.meshgrid(kr_x, kr_y) # la malla
# Puntos de interpolacion
#kx_1=kx-2*k[int((len(k)+1)/2)] # Frecuencias desplazadas, ojo:len(k) tiene que ser impar
kxx=kx.reshape(1,Np*Nf)[0]
kyy=ky.reshape(1,Np*Nf)[0]

F2_real=sc.griddata((kxx, kyy), F1_m.real,(kr_xx,kr_yy), method='linear', fill_value=0)
F2_complex=sc.griddata((kxx, kyy), F1_m.imag,(kr_xx,kr_yy), method='linear', fill_value=0)
F2=F2_real+1j*F2_complex # Funcion interpolada

# Grafica de la zona interpolada sin translacion(formato de imagen)
dF.plotImage(F2, x_min=kr_x.min(), x_max=kr_x.max(), y_min=kr_y.min(), y_max=kr_x.max(),xlabel_name='kx(1/m)', ylabel_name='ky(1/m)', title_name='Gráfica despues de la interpolación - [xo,yo]=[%i,%i]'%(Rt_x, Rt_y))

# Grafica en 3D de la zona interpolada(Parte real)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kr_xx,kr_yy,F2_real)
ax.scatter(kxx,kyy,F1_m.real,c='r',marker='.', label='Interpolation points')
ax.set_xlabel('kx(1/m)')
ax.set_ylabel('ky(1/m)')
ax.set_zlabel('Magnitud')
ax.set_title('Grafica de la parte real interpolada')
ax.legend()

#%%---------------CUARTA ETAPA: 2D-IFFT--------------------
#                  F(kx,ky) --> f(x,y)
#----------------------------------------------------------
#ifft_len = [len(F2), len(F2[0])]
#f3 = np.fliplr(np.rot90(np.fft.ifft2(F2, ifft_len)))
#window = 1.0 + np.hanning(len(Ky_even))
#F2 *= window

F3_aux=np.fft.ifftshift(F2,axes=(0,1)) # Shift antes del 2D-IFFT
F3= np.fft.ifft2(F3_aux)
F3= np.fft.fftshift(F3,axes=(0,1))

#dF.plotImage(F3, x_min=0, x_max=Lx, y_min=0, y_max=Ly,xlabel_name='Range(m)', ylabel_name='Cross-Range(m)', title_name='Image of Ro=[%i,%i]m'%(Rt_x,Rt_y))
# Grafica 
cmap="plasma"
fig, ax = plt.subplots()
im=ax.imshow(20*np.log10(abs(F3)),cmap,origin='lower', extent=[0, Lx, 0, Ly])#vmin=-600,vmax=-100)
ax.set(xlabel='Range(m)',ylabel='Cross-range(m)', title='Image of Ro=[%i,%i]m'%(Rt_x,Rt_y))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
plt.colorbar(im,cax=cax,label='Reflectividad(dB)',extend='both')
