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

# Molecule file
#molfile = '/home/jujuman/Research/MD_TEST/Chignolin/1uao_H.pdb'
#molfile = '/home/jujuman/Research/IR_MD/M3/m3.xyz'
#molfile = '/home/jujuman/Research/Opt_test/1d.pdb'
molfile = '/home/jujuman/Research/MD_TEST/C_2500/C_2500.xyz'

# Dynamics file
#xyzfile = '/home/jujuman/Research/MD_TEST/Chignolin/mdcrd.xyz'
#xyzfile = '/home/jujuman/Research/IR_MD/M3/mdcrd.xyz'
#xyzfile = '/home/jujuman/Research/Opt_test/mdcrd_1d.xyz'
xyzfile = '/home/jujuman/Research/MD_TEST/C_2500/mdcrd.xyz'

# Trajectory file
#trajfile = '/home/jujuman/Research/MD_TEST/Chignolin/traj.dat'
#trajfile = '/home/jujuman/Research/IR_MD/M3/traj.dat'
#trajfile = '/home/jujuman/Research/Opt_test/traj_1d.dat'
trajfile = '/home/jujuman/Research/MD_TEST/C_2500/traj.dat'

# Optimized structure out
#optfile = '/home/jujuman/Research/MD_TEST/Chignolin/optmol.xyz'
#optfile = '/home/jujuman/Research/IR_MD/M3/optmol.xyz'
#optfile = '/home/jujuman/Research/Opt_test/optmol_1d.xyz'
optfile = '/home/jujuman/Research/MD_TEST/C_2500/optmol.xyz'

T = 2200.0 # Temperature
C = 8.0 # Optimization convergence

#wkdir    = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk/'
wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_2/cv4/'
#wkdir = '/home/jujuman/Research/ForceTrainTesting/train_full_al1/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + '/train'
Nn = 5
#nnfdir   = wkdir + 'networks/'

#----------------------------------------------

# Load molecule
mol = read(molfile)

L = 70.0
mol.set_cell(([[L, 0, 0],
               [0, L, 0],
               [0, 0, L]]))

mol.set_pbc((True, True, True))

# Set NC
aens = ensemblemolecule(cnstfile, saefile, nnfdir, Nn, 0)

# Set ANI calculator
mol.set_calculator(ANIENS(aens,sdmx=20000000.0))

# Optimize molecule
start_time = time.time()
dyn = LBFGS(mol)
dyn.run(fmax=C)
print('[ANI Total time:', time.time() - start_time, 'seconds]')

print(hdt.evtokcal*mol.get_potential_energy())


# Save optimized mol
spc = mol.get_chemical_symbols()
pos = mol.get_positions(wrap=True).reshape(1,len(spc),3)

hdt.writexyzfile(optfile, pos, spc)


#exit(0)

# Open MD output
mdcrd = open(xyzfile,'w')

# Open MD output
traj = open(trajfile,'w')

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 0.5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(mol, 0.2 * units.fs, T * units.kB, 0.05)

# Run equilibration
#print('Running equilibration...')
#start_time = time.time()
#dyn.run(50000) # Run 100ps equilibration dynamics
#print('[ANI Total time:', time.time() - start_time, 'seconds]')

# Set the momenta corresponding to T=300K
#MaxwellBoltzmannDistribution(mol, T * units.kB)
# Print temp
ekin = mol.get_kinetic_energy() / len(mol)
print('Temp: ', ekin / (1.5 * units.kB))

# Define the printer
def printenergy(a=mol, d=dyn, b=mdcrd, t=traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    #print()
    stddev =  hdt.evtokcal*a.calc.stddev

    t.write(str(d.get_number_of_steps()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' + str(ekin) + ' ' + str(epot+ekin) + '\n')
    b.write(str(len(a)) + '\n' + str(ekin / (1.5 * units.kB)) + ' Step: ' + str(d.get_number_of_steps()) + '\n')
    c = a.get_positions(wrap=True)
    for j, i in zip(a, c):
        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' ' StdDev = %.3fKcal/mol/atom' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))

# Attach the printer
dyn.attach(printenergy, interval=10)

# Run production
print('Running production...')
start_time = time.time()
dyn.run(15000) # Do 0.5ns of MD

for i in range(100):
    dyn.set_temperature(300.0 * units.kB)
    dyn.run(15000)  # Do 0.5ns of MD
    dyn.set_temperature(T * units.kB)
    dyn.run(15000)  # Do 0.5ns of MD

dyn.set_temperature(300.0 * units.kB)
dyn.run(50000) # Do 0.5ns of MD
print('[ANI Total time:', time.time() - start_time, 'seconds]')
mdcrd.close()
traj.close()
print('Finished.')