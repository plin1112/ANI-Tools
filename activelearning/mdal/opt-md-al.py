import pyaniasetools as pya
import gdbsearchtools as gdb

import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import random

import os

#--------------Parameters------------------
wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv2/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

At = ['C', 'O', 'N'] # Hydrogens added after check

dstore = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/confs_3/'

N = 2
T = 800.0
dt = 0.5

idir = [(0.50,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_1/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_2/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/inputs/'),
        (0.60,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_1/inputs/'),
        (0.50,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_2/inputs/'),
        (0.50,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_3/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_1/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_2/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_3/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_4/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_1/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_2/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_3/inputs/'),
        (0.30,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_4/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s02/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s03/inputs/'),
        (0.35,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s04/inputs/'),
        (0.35,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s05/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_03_red/inputs/'),
        (1.00,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_04_red/inputs/'),
        (0.50,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_05_red/inputs/'),
        (0.35,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_06_red/inputs/'),
        ]

#-------------------------------------------

activ = pya.moldynactivelearning(cnstfile, saefile, wkdir+'train', 5)

difo = open(dstore + 'info_data_mdso.nfo', 'w')
Nmol = 0
for di,id in enumerate(idir):
    files = os.listdir(id[1])
    random.shuffle(files)

    dnfo = str(di) + ' of ' + str(len(idir)) + ') dir: ' + str(id) + ' Selecting: '+str(id[0]*len(files))
    print(dnfo)
    Nmol += len(files)
    difo.write(dnfo+'\n')
    for n,m in enumerate(files[0:int(id[0]*len(files))]):
        data = hdn.read_rcdb_coordsandnm(id[1]+m)
        S =  data["species"]
        print(n,') Working on',m,'...')

        # Set mols
        activ.setmol(data["coordinates"], S)

        # Generate conformations
        X = activ.generate_conformations(N, T, dt, 700, 10, dS = 0.2)

        nfo = activ._infostr_
        difo.write('  -'+m+': '+nfo+'\n')
        difo.flush()
        print(nfo)

        if X.size > 0:
            hdn.writexyzfile(dstore+'mds_'+m.split('.')[0]+'_'+str(di).zfill(4)+'.xyz',X,S)
print(Nmol)
difo.close()
