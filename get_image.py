from numpy import *
from matplotlib.pyplot import *
import glob
import os
from scipy.stats import mode
from scipy import ndimage
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
matplotlib.rcParams.update({'font.size': 18})




#berglocs=load('20120816_172400_20120816_182200.npy')
def plotarrows(filename):
  clf()
  bgm=imread(filename)
  figname=os.path.basename(filename)[0:-4]
  figure()
  #make background black
  ax=subplot('111', axisbg='black')
  
  #title(figname) #creates full figure name, doesn't look good for publishing figures
  
  #get figure title from filename, works only for this dataset
  a=figname
  figtitle='Aug. '+a[6:8]+' '+a[9:11]+':'+a[11:13]
  title(figtitle)
 
  imshow((bgm),origin='upper',alpha=1,zorder=3,cmap='gray')
  
  #to plot north arrow
  ax.arrow(10,310,28,25,head_length=10,head_width=10, fc='w',ec='w',zorder=5)
  text(52,356,'N',fontsize=16,color='w',zorder=6)
  
  #change labels to show proper distances
  xlabel('Distance (km)')
  ylabel('Distance (km)')
  ax.xaxis.set_ticks([0,250,500])
  ax.set_xticklabels(['0','2.5','5'])
  ax.yaxis.set_ticks([300,500,700])
  ax.set_yticklabels(['0','2','4'])
  
  axis([0,500,700,300])
  #savefig('arrow/'+figname+'.png')
  savefig('arroworig/'+figname+'.png',dpi=300)
  #savefig('figs/arrow'+figname+'.png')

  return
  


#read all npy files hour by hour
directory='img/'
filetype='*3400*.png'
#must do this fo 0400 and 3400
files=glob.glob(os.path.join(directory+filetype))
files.sort()


for i in range(len(files)):
  print i
  plotarrows(files[i])

