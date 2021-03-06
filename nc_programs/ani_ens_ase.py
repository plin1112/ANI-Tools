from ase_interface import ANIENS
from ase_interface import ensemblemolecule

import pyNeuroChem as pync
import pyaniasetools as pya
import hdnntools as hdt

import numpy as np
import  ase
import time
#from ase.build import molecule
#from ase.neb import NEB
#from ase.calculators.mopac import MOPAC
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.optimize.fire import FIRE as QuasiNewton

from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger

#from ase.neb import NEBtools
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

import matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

#----------------Parameters--------------------

dir = '/home/jujuman/scratch/MDTest/2avi/'


# Molecule file
molfile = dir + '2avi_solv.pdb'

# Dynamics file
xyzfile = dir + 'mdcrd.xyz'

# Trajectory file
trajfile = dir + 'traj.dat'

# Optimized structure out:
optfile = dir + 'optmol.xyz'

T = 300.0 # Temperature
dt = 0.25
C = 0.1 # Optimization convergence
steps = 400000

wkdir = '/home/jsmith48/Gits/ANI-Networks/networks/al_networks/ANI-AL-0808.0303.0400/'
cnstfile = wkdir + 'train0/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'train0/sae_wb97x-631gd.dat'
nnfdir   = wkdir + '/train'

Nn = 5
#nnfdir   = wkdir + 'networks/'

# Load molecule
mol = read(molfile)
#print('test')
L = 70.0
mol.set_cell(([[L, 0, 0],
               [0, L, 0],
               [0, 0, L]]))

mol.set_pbc((True, True, True))

#print(mol.get_chemical_symbols())

# Set NC
aens = ensemblemolecule(cnstfile, saefile, nnfdir, Nn, 5)

# Set ANI calculator
mol.set_calculator(ANIENS(aens,sdmx=20000000.0))

print(mol.get_potential_energy())
print(np.where(mol.get_forces()>200.0))

print("size: ", len(mol.get_chemical_symbols()))

# Optimize molecule
start_time = time.time()
dyn = QuasiNewton(mol)
dyn.run(fmax=C)
print('[ANI Total time:', time.time() - start_time, 'seconds]')

#print(hdt.evtokcal*mol.get_potential_energy())
#print(hdt.evtokcal*mol.get_forces())

# Save optimized mol
spc = mol.get_chemical_symbols()
pos = mol.get_positions(wrap=False).reshape(1,len(spc),3)

hdt.writexyzfile(optfile, pos, spc)

# Open MD output
mdcrd = open(xyzfile,'w')

# Open MD output
traj = open(trajfile,'w')

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 0.5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(mol, dt * units.fs, T * units.kB, 0.02)

# Run equilibration
#print('Running equilibration...')
#start_time = time.time()
#dyn.run(10000) # Run 100ps equilibration dynamics
#print('[ANI Total time:', time.time() - start_time, 'seconds]')

# Set the momenta corresponding to T=300K
#MaxwellBoltzmannDistribution(mol, T * units.kB)
# Print temp
#ekin = mol.get_kinetic_energy() / len(mol)
#print('Temp: ', ekin / (1.5 * units.kB))

# Define the printer
def storeenergy(a=mol, d=dyn, b=mdcrd, t=traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    stddev =  hdt.evtokcal*a.calc.stddev

    t.write(str(d.get_number_of_steps()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' + str(ekin) + ' ' + str(epot+ekin) + '\n')
    b.write(str(len(a)) + '\n' + str(ekin / (1.5 * units.kB)) + ' Step: ' + str(d.get_number_of_steps()) + '\n')
    c = a.get_positions(wrap=True)
    for j, i in zip(a, c):
        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' ' StdDev = %.3fKcal/mol/atom' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))

# Define the printer
def printenergy(a=mol, d=dyn, b=mdcrd, t=traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    stddev =  hdt.evtokcal*a.calc.stddev

    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' ' StdDev = %.3fKcal/mol/atom' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))


# Attach the printer
dyn.attach(storeenergy, interval=250)
#dyn.attach(printenergy, interval=1)

# Run production
print('Running production...')
#start_time = time.time()
#for i in range(int(T)):
#    print('Set temp:',i,'K')
#    dyn.set_temperature(float(i) * units.kB)
#    dyn.run(50)

mol.calc.nc_time = 0.0

dyn.set_temperature(T * units.kB)
start_time = time.time()
dyn.run(steps)
final_time = time.time() - start_time
print('[NeuroChem time:', mol.calc.nc_time, 'seconds]')
print('[Total time:', final_time, 'seconds]')
mdcrd.close()
traj.close()
print('Finished.')
