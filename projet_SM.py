import numpy as np 
import scipy.fft as sc
import matplotlib.pyplot as plt
from matplotlib import cm

def calcul_u(nu = 1, t_max = 200, L = 100 , N= 1024, delta_t = 0.05, Compute = True, Plot = True):

    M = int(t_max/delta_t)+1

    t = np.linspace(0, t_max, M)

    u = np.zeros((M,N))
    u_chap = np.zeros((M,N), dtype = 'complex')


    k = np.arange(-N/2 , N/2)


    F_L = (2*np.pi*k /L)**2 - nu * (2*np.pi*k/L)**4

    x = np.linspace(0,L-L/N, N)
    
    if Compute == True:

        u[0,:]= np.cos(2*np.pi* x /L) +0.1* np.cos(4*np.pi* x/L)
        u[1,:]= np.cos(2*np.pi* x /L) +0.1* np.cos(4*np.pi* x/L)

        u_chap[0,:] = sc.fftshift(sc.fft( u[0,:]))
        u_chap[1,:] = sc.fftshift(sc.fft(( u[1,:])))


        i= complex(0,1)
    
        C_1 = (1+delta_t/2* F_L) / (1-delta_t/2* F_L)
        C_2 = i *np.pi *delta_t*k/L
        C_3 = 1 - delta_t*F_L/2
    
    
    
        for n in range (1,M-1):
        
        
            u_chap[n+1, :] = C_1 * u_chap[n,:] - C_2*( 3/2 * sc.fftshift(sc.fft( u[n]**2 )) - 1/2 * sc.fftshift(sc.fft( u[n-1]**2 )) ) / C_3
        
            u[n+1,:] = np.real(sc.ifft(sc.ifftshift( u_chap[n+1,:])))
            
        if Plot == True :
            #calcul_u (compute = False)
            xx, tt = np.meshgrid(x,t)

            plt.show()
            plt.figure()
            

            fig, ax = plt.subplots()
            c = np.linspace(np.min(np.min((u))), np.max(np.max((u))),101)
            im=ax.contourf(xx,tt, u, c, cmap = cm.jet)
            plt.colorbar(im)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.title(f"Kuramoto-Sivashinsky viscosité = {nu}")
            plt.show()
        
            return u, N, nu
        else :
            return u, N, nu
    else : 
        return  

    

liste_L1= np.arange(1,50,0.5)
liste_L2 = np.arange(1,17, 0.25)

def amplitude (valeurs_L, nu) :
    
    A = []
    for L in valeurs_L:
        u, N, nu = calcul_u(L= L,nu = nu,t_max = 500 ,Plot = False)
        A.append( np.sqrt(1/L * sum(u[-1]**2) )/N)
        
    plt.figure()
    
    plt.plot(valeurs_L, A)
    plt.xlabel("L")
    plt.ylabel("A")
    plt.title(f"Amplitude de u selon la valeurs de L avec ν = {nu}")
        

# Voici les différentes lignes de lancement pour obtenir tous les graphes qui sont dans le rapport :

# La figure qu'on devait refaire
calcul_u() 

# Les plot qui donnent la même figure que celle qu'on devait reproduire mais en changeant la viscosité
calcul_u(nu =2)
calcul_u(nu =3)

# Les graphiques mettant en avant le fait qu'en dessous de L_critique, la solution tend vers 0

# pour nu =1
calcul_u(nu = 1, L = 6)
calcul_u(nu = 1, L = 6.25)

# pour nu =2
calcul_u(nu = 2, L = 8.25)
calcul_u(nu = 2, L = 8.75)

# pour nu =3
calcul_u(nu = 3, L = 10.25)
calcul_u(nu = 3, L = 10.75)

# On va refaire les 4 figures:
    
# Figure 1
calcul_u(nu = 1, L = 10)

# Figure 2 
calcul_u(nu =1, L =20)

# Figure 3
calcul_u(nu =1, L =30)

# Figure 4
calcul_u(nu =1, L =40)


# Mise en avant de la solution triviale pour L = 100

calcul_u(nu =1000, L =100)


# Les graphiques d'amplitudes :
    
# pour nu = 1
amplitude(liste_L1, 1)

# pour nu = 2
amplitude(liste_L1, 2)
amplitude(liste_L2, 2)

# pour nu = 3
amplitude(liste_L1, 3)
amplitude(liste_L2, 3)
