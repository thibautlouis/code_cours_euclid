# Copy of lewis 4d plot notebook
# https://github.com/cmbant/notebooks/blob/master/Growth-Of-Structure-21_CMB.ipynb

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots

# some options
DES=False
TTTEEE=True
BAO=True
shadows=True
colorbar=True
animate=True
export_mp4=False
filename =  "test.mp4"

priors = 'lenspriors'

omm_lim=(0.1,0.8)
H0_lim=(40,100)
s8_lim=(0.6,1)


if DES:
    omm_lim=(0.1,0.8)
    H0_lim=(40,100)
    s8_lim=(0.6,1.2)
    
if priors == 'DESpriors':
    H0_lim=(55,91)
    omm_lim=(0.1,0.9)
    alpha=0.3
    thin=1


# using public Planck chain grid, change path appropriately (get the file from the PLA)
g = plots.get_single_plotter(chain_dir="/Users/thibautlouis/Desktop/projects/Planck/COM_CosmoParams_fullGrid_R3.01/", width_inch=8)

# define a custom derived parameter, here theta_BAO in terms of rdrag and DM051 stored in chains
from getdist.paramnames import ParamInfo
class ThetaBAO(ParamInfo):
    def getDerived(self, par):
        return par.rdrag / par.DM051/np.pi*180

theta_bao = ThetaBAO(name='theta_BAO',label=r'\theta_{\rm BAO}(0.51) {\rm [degrees]}')


roots = ['base_lensing_%s'%priors]
if BAO:
    roots +=  ['base_lensing_BAO_%s'%priors]
if TTTEEE:
    roots +=  ['base_plikHM_TTTEEE_lowl_lowE']

g.plot_4d(roots, ['H0', 'omegam', 'sigma8', theta_bao], cmap='viridis', 
          alpha = [0.3,0.05,0.05], shadow_alpha=[0.1,0.01,0.03], colorbar_args={'shrink': 0.7},
          lims={'H0':(40,100), 'omegam':(0.1, 0.8)}, shadow_color=shadows, 
          compare_colors=['C1','k'], animate=animate, anim_fps=20, mp4_filename=filename)
