#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jekosch/radmann/schneehagen
optimize plate silencer dimensions
"""



# function to optimize
# optimizes for 4 chambers, schneehagen
def opti_Lc(geo, *f, diss=0):
	
    #hier kannst du noch parameter über *f übergeben		
    Hd = 0.06
    dd = 0.08
    mu = 0.4
    Hc=f[1]
    eta=0.01
    #eta=eta
    # das müsstest du dann anpassen, fall du zum Beispiel E oder rho auch optimieren willst
    ind=list(range(len(geo)))
    ind.pop(1) # hier wird nur der eintrag für die höhe entfernt und der
               #rest sind dann nur noch die längen die dann den cavities übergeben werden
    Lc=geo[ind]
    Hp=geo[1]
  	
    #der Teil gibt verscheidene Möglichkeiten für die Optimierung der Breitbandigkeit: 
    # plusminus 70 Hz oder Prozent ...		
    if diss==0: #else just use the frequencies provided
        f=f[0]
        f = np.linspace(f-0.1*f, f+0.1*f, num=15) #modespm10
        #f = np.linspace(f-0.1*f, f+0.1*f, num=5) #modespm10_5f
        #f = np.linspace(f-70, f+70, num=5) #modes_5f
        #f=np.append(f, 1000.)
    temp1 = Temperature(C=25)
    
    fluid1 = Fluid(temperature=temp1)
    material1= Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*eta))
    
    plate1 = SimplePlate3D(hp = Hp, material = material1, temperature = temp1)

    # mode orders
    # number of modes 3D
    J = np.ceil(np.sqrt(np.max(f)*2*np.max(Lc)**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    K = np.ceil(np.sqrt(np.max(f)*2*dd**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    R = np.ceil(2*np.max(f)*np.max(Lc)/np.real(fluid1.c))
    S = np.ceil(2*np.max(f)*dd/np.real(fluid1.c))
    T = np.ceil(2*np.max(f)*Hc/np.real(fluid1.c))
    
    # plate modes
    j = np.arange(1, J+1, 1, dtype='int')
    k = np.arange(1, K+1, 1, dtype='int')
    l = np.arange(1, J+1, 1, dtype='int')
    n = np.arange(1, K+1, 1, dtype='int')
    
    # cavity modes
    r = np.arange(0, R+J, 1, dtype='int')
    s = np.arange(0, S+K, 1, dtype='int')
    t = np.arange(0, 5*T, 1, dtype='int')
    
    # jede kammer hat das gleiche Plattenmaterial,
    #  und in 'ind' stehen dann nur noch die anzahl der kammern,
    #  die dann hier durch iteriert werden
    cavity = Cavity3D(height = Hc, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)

    lining = [SinglePlateResonator3D(length=Lc[i], depth=dd,
                                     j=j, k=k, l=l, n=n, t=t, plate=plate1,
                                     cavity=cavity) for i in range(len(ind))]
    ductelement = [DuctElement3D(lining=lining[i], medium=fluid1,
                                  M=0.0) for i in range(len(ind))]
    
    duct = Duct3D(freq=f, height_d=Hd, elements=list(ductelement))
    
    #hier wird das optimierungsziel festgelegt. der Wert 'opti'
    # soll minimal werden. In diesem Fall die der 10log der 
    # Summe der Transmissiongrade (aus dem festgelegten Frequenzbereich) 
    # geteilt durch die Anzahl der Frequenzstützstellen, für die optimiert wurde.
    tra1, ref1, dis1 = duct.scatteringcoefficients('forward')
    tra=sum(tra1)
    opti=10*np.log10(tra/len(f))#+10*np.log10(dis/len(f[:-1]))
     #white noise with amplitude 1: (tra*1)/(1)

    return opti



##

def opti_fix_size(geo, *f, diss=0):
	
    #hier kannst du noch parameter über *f übergeben		
    Hd = 0.06
    dd = 0.08
    mu = 0.4
    Hc=f[1]
    eta=0.01
    #eta=eta
    # das müsstest du dann anpassen, fall du zum Beispiel E oder rho auch optimieren willst
    ind=list(range(len(geo)))
    ind.pop(1) # hier wird nur der eintrag für die höhe entfernt und der
               #rest sind dann nur noch die längen die dann den cavities übergeben werden
    Lc=geo[ind]
    Hp=geo[1]
  	
    #der Teil gibt verscheidene Möglichkeiten für die Optimierung der Breitbandigkeit: 
    # plusminus 70 Hz oder Prozent ...		
    if diss==0: #else just use the frequencies provided
        f=f[0]
        f = np.linspace(f-0.1*f, f+0.1*f, num=15) #modespm10
        #f = np.linspace(f-0.1*f, f+0.1*f, num=5) #modespm10_5f
        #f = np.linspace(f-70, f+70, num=5) #modes_5f
        #f=np.append(f, 1000.)
    temp1 = Temperature(C=25)
    
    fluid1 = Fluid(temperature=temp1)
    material1= Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*eta))
    
    plate1 = SimplePlate3D(hp = Hp, material = material1, temperature = temp1)

    # mode orders
    # number of modes 3D
    J = np.ceil(np.sqrt(np.max(f)*2*np.max(Lc)**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    K = np.ceil(np.sqrt(np.max(f)*2*dd**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    R = np.ceil(2*np.max(f)*np.max(Lc)/np.real(fluid1.c))
    S = np.ceil(2*np.max(f)*dd/np.real(fluid1.c))
    T = np.ceil(2*np.max(f)*Hc/np.real(fluid1.c))
    
    # plate modes
    j = np.arange(1, J+1, 1, dtype='int')
    k = np.arange(1, K+1, 1, dtype='int')
    l = np.arange(1, J+1, 1, dtype='int')
    n = np.arange(1, K+1, 1, dtype='int')
    
    # cavity modes
    r = np.arange(0, R+J, 1, dtype='int')
    s = np.arange(0, S+K, 1, dtype='int')
    t = np.arange(0, 5*T, 1, dtype='int')
    
    # jede kammer hat das gleiche Plattenmaterial,
    #  und in 'ind' stehen dann nur noch die anzahl der kammern,
    #  die dann hier durch iteriert werden
    cavity = Cavity3D(height = Hc, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)

    lining = [SinglePlateResonator3D(length=Lc[i], depth=dd,
                                     j=j, k=k, l=l, n=n, t=t, plate=plate1,
                                     cavity=cavity) for i in range(len(ind))]
    ductelement = [DuctElement3D(lining=lining[i], medium=fluid1,
                                  M=0.0) for i in range(len(ind))]
    
    duct = Duct3D(freq=f, height_d=Hd, elements=list(ductelement))
    
    #hier wird das optimierungsziel festgelegt. der Wert 'opti'
    # soll minimal werden. In diesem Fall die der 10log der 
    # Summe der Transmissiongrade (aus dem festgelegten Frequenzbereich) 
    # geteilt durch die Anzahl der Frequenzstützstellen, für die optimiert wurde.
    tra1, ref1, dis1 = duct.scatteringcoefficients('forward')
    tra=sum(tra1)
    opti=10*np.log10(tra/len(f))#+10*np.log10(dis/len(f[:-1]))
     #white noise with amplitude 1: (tra*1)/(1)

    return opti



