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
  berglocs=load(filename)

  #pixel spacing in meters
  pixelspacing=10.
  #time difference between first and last measurement, minutes
  timediff=30.

  #find total displacement between first and last measurement
  dsdt=berglocs[-1]-berglocs[0]
  displacement=hypot(dsdt[:,0],dsdt[:,1])
  disp_criterion=2
  #number of pixels to move to consider to be have motion to prevent stationary icebergs from clogging data
  good_indices=where(displacement>disp_criterion)[0]
  berglocs_good=berglocs[:,good_indices,:]

  #this finds the start and end points for each iceberg and plots them as straigh linesa
  Y1=berglocs_good[0,:,0]
  X1=berglocs_good[0,:,1]
  Y2=berglocs_good[-1,:,0]
  X2=berglocs_good[-1,:,1]

  #get motion in x and y directions
  DY=Y2-Y1
  DX=X2-X1

  #convert VX and VY to cm/s
  VX=DX*100.*pixelspacing/(timediff*60.)
  VY=DY*100.*pixelspacing/(timediff*60.)

  nxcells=25
  nycells=25
  grid_x, grid_y = np.mgrid[xmin:xmax:complex(nxcells), ymin:ymax:complex(nycells)]

  vx_rbf=Rbf(X1,Y1,VX,function='linear')
  vxint=vx_rbf(grid_x.flatten(),grid_y.flatten())
  vxint=vxint.reshape([nxcells,nycells])
  vy_rbf=Rbf(X1,Y1,VY,function='linear')
  vyint=vy_rbf(grid_x.flatten(),grid_y.flatten())
  vyint=vyint.reshape([nxcells,nycells])
  
  


  ##show interpolated maps
  #figure()
  #subplot(1,2,1)
  #title('vx')
  #imshow(vxint,extent=[grid_x[:,0].min(),grid_x[:,0].max(),grid_y[0].min(),grid_y[0].max()])
  #colorbar()
  #subplot(1,2,2)
  #title('vy')
  #imshow(vyint,extent=[grid_x[:,0].min(),grid_x[:,0].max(),grid_y[0].min(),grid_y[0].max()])
  #colorbar()
  #show()
  
  figname=os.path.basename(filename)[0:-4]
  
  print 'velocity range '+figname
  fullvel=sqrt(vyint**2+vxint**2)
  print fullvel.min(),fullvel.max()
  
  figure()
  #make background black
  ax=subplot('111', axisbg='black')
  
  #title(figname) #creates full figure name, doesn't look good for publishing figures
  
  #get figure title from filename, works only for this dataset
  a=figname
  figtitle='Aug. '+a[6:8]+' '+a[9:11]+':'+a[11:13]+' - '+'Aug. '+a[22:24]+' '+a[25:27]+':'+a[27:29]
  title(figtitle)
  
  #get velocity of 1 pixel in cm/s
  pixelVel=100.*pixelspacing/(timediff*60.)
  arrowSize=18.
  arrowSpeed=arrowSize*pixelVel
  arrowLabel='%d'%(arrowSpeed)+' cm/s'
  #print arrowSize
  q=quiver(grid_x,grid_y,vxint,vyint,scale_units='xy', angles='xy', scale=.25, color='w',zorder=2)
  #p=quiverkey(q,0.0*grid_x.max(),1.51*grid_y.max(),arrowSpeed,arrowLabel,coordinates='data',color='r', labelpos='S',zorder=2)
  #p=quiverkey(q,50,350,arrowSpeed,arrowLabel,coordinates='data',color='r', labelpos='S',zorder=4)

  #imshow(bg,origin='upper',cmap='binary',zorder=1,interpolation='none')
  #zorder controls plotting order, blot radar amplitude background
  imshow((bgm),origin='upper',alpha=1,zorder=3,cmap='gray')
  
  #trick the quiverkey labeling to allow masked window for the figures by drawing a nan-valued quiverplot
  q1=quiver(grid_x,grid_y,nan*vxint,nan*vyint,scale_units='xy', angles='xy', scale=.25, color='w',zorder=4) #scale should be .4
  p=quiverkey(q1,50,650,arrowSpeed,arrowLabel,coordinates='data',color='w', labelpos='S',labelcolor='w')
  
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
  savefig('arrow/'+figname+'.png',dpi=300)
  #savefig('figs/arrow'+figname+'.png')

  return DX,DY,VX,VY,vxint,vyint
  




#read in mask
bg=imread('2012maskT.png')
bg=bg*bg.astype('bool')
naz,nr=shape(bg)
yvl,xvl=where(bg==1.0) #find mask boundaries
ymin=yvl.min()
ymax=yvl.max()
xmin=xvl.min()
xmax=xvl.max()

#make mask to plot overlays
bgm=imread('bgm1.png')
#get rid of noise
bgm[where(bgm<.3)]=0.0
#mask out lagoon
bgm[where(bg==1)]=nan
#imshow(bgm)
#show()


#read all npy files hour by hour
directory='recscipy/'
filetype='*.npy'
files=glob.glob(os.path.join(directory+filetype))
files.sort()

#files=['data/20120816_192400_20120816_202200.npy']
#files=['data/20120817_202400_20120817_212200.npy']

#files[32] is comparison figure

for i in range(len(files)):
  print i
  dx,dy,vx,vy,vxint,vyint=plotarrows(files[i])





