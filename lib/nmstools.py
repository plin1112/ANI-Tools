import numpy as np

mDynetoMet = 1.0e-5 * 1.0e-3 * 1.0e10
Kb = 1.38064852e-23
MtoA = 1.0e10

class nmsgenerator():
    # xyz = initial min structure
    # nmo = normal mode displacements
    # fcc = force constants
    # spc = atomic species list
    # T   = temperature of displacement
    def __init__(self,xyz,nmo,fcc,spc,T,minfc = 1.0E-3,maxd=2.0):
        self.xyz = xyz
        self.nmo = nmo
        self.fcc = np.array([i if i > minfc else minfc for i in fcc])
        self.chg = np.array([self.__getCharge__(i) for i in spc])
        self.T = T
        self.Na = xyz.shape[0]
        self.Nf = nmo.shape[0]
        self.maxd = maxd

    # returns atomic charge
    def __getCharge__(self,type):
        if type is 'H':
            return 1.0
        elif type is 'C':
            return 6.0
        elif type is 'N':
            return 7.0
        elif type is 'O':
            return 8.0
        elif type is 'F':
            return 9.0
        elif type is 'S':
            return 16.0
        elif type == 'Cl':
            return 17.0
        else:
            print('Unknown atom type! ',type)
            exit(1)

    # Checks for small atomic distances
    def __check_atomic_distances__(self,rxyz):
        for i in range(0,self.Na):
            for j in range(i+1,self.Na):
                Rij = np.linalg.norm(rxyz[i]-rxyz[j])
                #print(0.006 * (self.chg[i] * self.chg[j]) + 0.65)
                if Rij < 0.006 * (self.chg[i] * self.chg[j]) + 0.6:
                    #print('DIST:',self.chg[i],self.chg[j],0.006 * (self.chg[i] * self.chg[j]) + 0.6,Rij)
                    return True
        return False

    # Checks for large changes in atomic distance from eq
    def __check_distance_from_eq__(self,rxyz):
        for i in range(0,self.Na):
            Rii = np.linalg.norm(self.xyz[i]-rxyz[i])
            if Rii > 2.0:
                #print(Rii)
                return True
        return False

    # Generate a structure
    def __genrandomstruct__(self):
        rdt = np.random.random(self.Nf+1)
        rdt[0] = 0.0
        norm = np.random.random(1)[0]
        rdt = norm*np.sort(rdt)
        rdt[self.Nf] = norm

        oxyz = self.xyz.copy()

        for i in range(self.Nf):
            Ki = mDynetoMet * self.fcc[i]
            ci = rdt[i+1]-rdt[i]
            Sn = -1.0 if np.random.binomial(1,0.5,1) else 1.0
            Ri = Sn * MtoA * np.sqrt((3.0 * ci * Kb * float(self.Nf) * self.T)/(Ki))
            Ri = min([Ri,self.maxd])
            oxyz = oxyz + Ri * self.nmo[i]
        return oxyz

    # Call this to return a random structure
    def get_random_structure(self):
        gs = True
        fnd = True
        count = 0
        while gs:
            rxyz = self.__genrandomstruct__()
            gs = self.__check_atomic_distances__(rxyz) or self.__check_distance_from_eq__(rxyz)
            if count > 100:
                fnd=False
                break
            count += 1
        return rxyz,fnd

    # Call this to return a random structure
    def get_Nrandom_structures(self, N):
        a_xyz = np.empty((N,self.Na,3),dtype=np.float32)
        a_bool = np.empty((N),dtype=np.bool)
        for i in range(N):
            a_xyz[i],a_bool[i] = self.get_random_structure()
        return a_xyz[a_bool]

    # Generate a structure using random distribution
    def __genrandomstruct2__(self, norm=None):
        from numpy.random import default_rng
        rng = default_rng()

        # rdt = np.random.random(self.Nf+1)
        rdt = rng.random(self.Nf+1)
        rdt[0] = 0.0
        # norm = np.random.normal()
        if norm == None:
            norm = rng.random(1.e-6, 1.0)
        rdt = norm*np.sort(rdt)
        rdt[self.Nf] = norm
        dir_set = [-1.0, 1.0]
        # rnd = np.random.random(self.Nf)

        oxyz = self.xyz.copy()

        for i in range(self.Nf):
            Ki = mDynetoMet * self.fcc[i]
            ci = rdt[i+1]-rdt[i]
            # Sn = -1.0 if np.random.binomial(1,0.5,1) else 1.0
            Sn = np.random.choice(dir_set)
            Ei = 0.5 * Kb * float(self.Nf) * self.T * ci
            Ri = Sn * MtoA * np.sqrt((2.0 * Ei)/Ki)
            oxyz = oxyz + Ri * self.nmo[i]
        return oxyz

    # Call this to return a random structure using random distribution with extra check
    def get_random_structure2(self):
        gs = True
        fnd = True
        count = 0
        while gs:
            rxyz = self.__genrandomstruct2__()
            gs = self.__check_atomic_distances__(rxyz) or self.__check_distance_from_eq__(rxyz)
            if count > 100:
                fnd=False
                break
            count += 1
        return rxyz,fnd

    # Call this to return a number of random structure using random distribution without checking
    def get_Nrandom_structures2(self, N):
        from numpy.random import default_rng
        rng = default_rng()

        a_xyz = np.empty((N,self.Na,3),dtype=np.float32)
        norm_values = rng.uniform(size=N)
        norm_values[0] = max(norm_values[0], 1.e-6)
        for i in range(N):
            a_xyz[i] = self.__genrandomstruct2__(norm=norm_values[i])
        return a_xyz

    # Call this to return a random structure using normal distribution without checking
    def get_Nrandom_structures3(self, N):
        from scipy.stats import truncnorm
        # from math import sqrt

        a_xyz = np.empty((N,self.Na,3),dtype=np.float32)
        sigma = 1.0
        norm_values = truncnorm.rvs(1.e-6/sigma, 1.0/sigma, loc=0., scale=sigma, size=N)
        for i in range(N):
            a_xyz[i] = self.__genrandomstruct2__(norm=norm_values[i])
        return a_xyz

class nmsgenerator_RXN():
    # xyz = initial min structure
    # nmo = normal mode displacements
    # fcc = force constants
    # spc = atomic species list
    # T   = temperature of displacement
    def __init__(self,xyz,nmo,spc,l_val,h_val,fcc=None,T=None,minfc = 1.0E-3):
        self.xyz = xyz
        self.nmo = nmo
        if fcc is not None:
            self.fcc = np.array([i if i > minfc else minfc for i in fcc])
        else:
            self.fcc = None
        self.chg = np.array([self.__getCharge__(i) for i in spc])
        if T is not None:
            self.T = T
        self.l_val = l_val
        self.h_val = h_val
        self.Na = xyz.shape[0]

    # returns atomic charge
    def __getCharge__(self,type):
        if type is 'H':
            return 1.0
        elif type is 'C':
            return 6.0
        elif type is 'N':
            return 7.0
        elif type is 'O':
            return 8.0
        else:
            print('Unknown atom type! ',type)
            exit(1)

    # Checks for small atomic distances
    def __check_atomic_distances__(self,rxyz):
        for i in range(0,self.Na):
            for j in range(i+1,self.Na):
                Rij = np.linalg.norm(rxyz[i]-rxyz[j])
                if Rij < 0.0075 * (self.chg[i] * self.chg[j]) + 0.65:
                    return True
        return False

    # Checks for large changes in atomic distance from eq
    def __check_distance_from_eq__(self,rxyz):
        for i in range(0,self.Na):
            Rii = np.linalg.norm(self.xyz[i]-rxyz[i])
            if Rii > 2.0:
                #print(Rii)
                return True
        return False

    # Generate a structure
    def __genrandomstruct__(self):
        ("generating randomstruct")
        oxyz = self.xyz.copy()
        Ri = np.random.uniform(low=self.l_val, high=self.h_val, size=None)             
        oxyz = oxyz + Ri * self.nmo
        return oxyz

    # Call this to return a random structure
    def get_random_structure(self):
        gs = True
        while gs:
            rxyz = self.__genrandomstruct__()
            gs = self.__check_atomic_distances__(rxyz) or self.__check_distance_from_eq__(rxyz)
        return rxyz

    # Call this to return a random structure
    def get_Nrandom_structures(self, N):
        a_xyz = np.empty((N,self.Na,3),dtype=np.float32)
        for i in range(N):
            a_xyz[i] = self.get_random_structure()
        return a_xyz
