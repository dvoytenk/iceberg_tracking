from numpy import *
from matplotlib.pyplot import *
import glob
import os
from scipy.stats import mode
from scipy import ndimage
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import scipy.stats
matplotlib.rcParams.update({'font.size': 18})

#this is not used, but can be used for interactive analysis
#def calcerrors():
  #def plotsecondary(data1,data2):
      #plt.figure(2)
      #plt.clf()
      #plt.subplot(1,2,1)
      #plt.hist(data1)
      #plt.subplot(1,2,2)
      #plt.hist(data2)
      #plt.show()
      
  #def impixshow(data1,data2):
    #fig = plt.figure(1)
    #plt.imshow(sqrt(data1**2+data2**2))
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    #plt.show()    
    
      
  #def onclick(event):
    #if event.button==3:
      #i=int(event.ydata)
      #j=int(event.xdata)
      #print i,j
      #vxmc=VXINT[:,i,j]
      #vymc=VYINT[:,i,j]
      #print 'kolmogorov-smirnov test given vxmc/vymc'
      ##the k-s test requires nonparametric values, so subtract out the mean, and divide by the standard deviation
      #print scipy.stats.kstest((vxmc-vxmc.mean())/vxmc.std(),'norm'),scipy.stats.kstest((vymc-vymc.mean())/vymc.std(),'norm')
      #plotsecondary(vxmc,vymc)
      ##global vxmc,vymc
      #return  

  ##read in mask
  #bg=imread('2012maskT.png')
  #bg=bg*bg.astype('bool')
  #naz,nr=shape(bg)
  #yvl,xvl=where(bg==1.0) #find mask boundaries
  #ymin=yvl.min()
  #ymax=yvl.max()
  #xmin=xvl.min()
  #xmax=xvl.max()

  ##make mask to plot overlays
  #bgm=imread('bgm1.png')
  ##get rid of noise
  #bgm[where(bgm<.3)]=0.0
  ##mask out lagoon
  #bgm[where(bg==1)]=nan
  ##imshow(bgm)
  ##show()

  #filename='recscipy/20120817_083400_20120817_090400.npy'

  #berglocs=load(filename)

  ##pixel spacing in meters
  #pixelspacing=10.
  ##time difference between first and last measurement, minutes
  #timediff=30.

  ##find total displacement between first and last measurement
  #dsdt=berglocs[-1]-berglocs[0]
  #displacement=hypot(dsdt[:,0],dsdt[:,1])
  #disp_criterion=2
  ##number of pixels to move to consider to be have motion to prevent stationary icebergs from clogging data
  #good_indices=where(displacement>disp_criterion)[0]
  #berglocs_good=berglocs[:,good_indices,:]

  ##this finds the start and end points for each iceberg and plots them as straigh linesa
  #Y1=berglocs_good[0,:,0]
  #X1=berglocs_good[0,:,1]
  #Y2=berglocs_good[-1,:,0]
  #X2=berglocs_good[-1,:,1]

  ###get motion in x and y directions
  ##DY=Y2-Y1
  ##DX=X2-X1

  ###convert VX and VY to cm/s
  ##VX=DX*100.*pixelspacing/(timediff*60.)
  ##VY=DY*100.*pixelspacing/(timediff*60.)

  #nxcells=25
  #nycells=25
  #grid_x, grid_y = np.mgrid[xmin:xmax:complex(nxcells), ymin:ymax:complex(nycells)]


  #VXINT=[]
  #VYINT=[]
  #pixerr=1.0

  #for i in range(1000):
    ##get motion in x and y directions
    #DY=((Y2+pixerr*random.randn(len(Y1)))-(Y1+pixerr*random.randn(len(Y1))))
    #DX=((X2+pixerr*random.randn(len(X1)))-(X1+pixerr*random.randn(len(X1))))
    ##DX=(X2-X1)+pixerr*random.randn(lden(X1))

    ##convert VX and VY to cm/s
    #VX=DX*100.*pixelspacing/(timediff*60.)
    #VY=DY*100.*pixelspacing/(timediff*60.)
    
    #vx_rbf=Rbf(X1,Y1,VX,function='linear')
    #vxint=vx_rbf(grid_x.flatten(),grid_y.flatten())
    #vxint=vxint.reshape([nxcells,nycells])
    #vy_rbf=Rbf(X1,Y1,VY,function='linear')
    #vyint=vy_rbf(grid_x.flatten(),grid_y.flatten())
    #vyint=vyint.reshape([nxcells,nycells])
    #VXINT.append(vxint)
    #VYINT.append(vyint)
    ##fig=figure()
    ##imshow(vxint)
    ##colorbar()
    ##savefig('intfigs/fg'+str(i)+'.png')
    ##close(fig)
    #print i
    
  #VXINT=array(VXINT)
  #VYINT=array(VYINT)

  #stdxint=zeros([nycells,nxcells])
  #stdyint=zeros([nycells,nxcells])
  #for i in range(nycells):
    #for j in range(nxcells):
      #stdxint[i,j]=std(VXINT[:,i,j])
      #stdyint[i,j]=std(VYINT[:,i,j])

  #return VXINT,VYINT



def make_ellipse(vxd,vyd,chisqval):
  evals,evecs=linalg.eig(cov(vxd,vyd))
  t=linspace(0,2*pi,100)
  xy=[cos(t),sin(t)]
  xyellipse=dot((sqrt(chisqval)*evecs*sqrt(evals)),xy) 
  xellipse=xyellipse[0]
  yellipse=xyellipse[1]
  return xellipse,yellipse


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
  
  #downsample for error ellipses
  gi=ones([nycells,nxcells])
  gi=gi*nan
  #this randomply picks data points, we provide our randomply picked ones as xrand and yrand.txt
  #yrand=random.random_integers(2,23,30)
  #xrand=random.random_integers(0,nxcells-1,30)
  #savetxt('xrand.txt',xrand)
  #savetxt('yrand.txt',yrand)
  xrand=loadtxt('xrand.txt')
  yrand=loadtxt('yrand.txt')
  gi[list(yrand),list(xrand)]=1.0
  xEllipsePoints=grid_x[list(yrand),list(xrand)]
  yEllipsePoints=grid_y[list(yrand),list(xrand)]
  #plot(xEllipsePoints,yEllipsePoints,'r*',markersize=10)
  VXINT,VYINT=calcerrors()

    
     
  #this is to plot the selected points
  #nn=quiver(grid_x,grid_y,vxint*gi,vyint*gi,scale_units='xy', angles='xy', scale=.25, color='w',zorder=2)

  q=quiver(grid_x,grid_y,vxint,vyint,scale_units='xy', angles='xy', scale=.25, color='w',zorder=2,pivot='tip')
  print "VXINTSIZE"
  print shape(vxint)
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


  for q in range(len(xrand)):
    xe,ye=make_ellipse(VXINT[:,yrand[q],xrand[q]],VYINT[:,yrand[q],xrand[q]],5.99)
    #scale=
    xen=4*xe+xEllipsePoints[q] #the 4 is related to the scale being .25 (one over scale)
    #print xen
    yen=4*ye+yEllipsePoints[q]
    #print xen,yen
    plot(xen,yen,'r',linewidth=2)
    #plot(140,600,'r+',markersize=20)

  axis([0,500,700,300])
  #savefig('arrow/'+figname+'.png')
  

  savefig('errell'+figname+'.png',dpi=300)
  
  
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


##read all npy files hour by hour
#directory='/home/denis/detection/2012/recscipy/'
#filetype='*.npy'
#files=glob.glob(os.path.join(directory+filetype))
#files.sort()

#files=['data/20120816_192400_20120816_202200.npy']
#files=['data/20120817_202400_20120817_212200.npy']

#files[32] is comparison figure


dx,dy,vx,vy,vxint,vyint=plotarrows('recscipy/20120817_083400_20120817_090400.npy')






##get ranges of min max values to interpolate
#ymin=berglocs_good[:,:,0].min()
#ymax=berglocs_good[:,:,0].max()
#xmin=berglocs_good[:,:,1].min()
#xmax=berglocs_good[:,:,1].max()






#arrowSize=.03
#arrowSpeed=10.
#arrowLabel='%d'%(arrowSpeed)+' cm/s'
#p=quiverkey(q,1.0*grid_x.max(),1.01*grid_y.max(),arrowSize,arrowLabel,coordinates='data',color='r')
#for k in range(0,shape(berglocs)[1]):
  #pl=[bl[k] for bl in berglocs]
  #y=[yx[0] for yx in pl]
  #x=[yx[1] for yx in pl]
  ##check to see if the mode value is the last value (stationary or not found berg)
  ##if mode(y)[0]!=y[-5]:
  ##plot(x,y,'*')
  #plot(x[0::samplingrate],y[0::samplingrate])
##axis([0,200,0,300])
#imshow(imread('2012mask.png'),origin='upper')

#samplingrate=10
#measurementRate=input('how many minutes per measurement? ') 
#pixelsize=10 #meters
#pixelVel=(pixelsize*100./(measurementRate*60.))/samplingrate#velocity of 1 pixel of motion in cm/s


#bergLocVel=[] #this stores the iceberg position and velocity info
#for k in range(0,shape(berglocs)[1]): #this iterates over the number of icebergs
  #pl=[bl[k] for bl in berglocs]
  #y=[yx[0] for yx in pl]
  #x=[yx[1] for yx in pl]
  ##check to see if the mode value is the last value (stationary or not found berg)
  ##if mode(y)[0]!=y[-5]:
  ##plot(x,y,'*')
  #x=x[0::samplingrate]
  #y=y[0::samplingrate]
  #vx=diff(x)
  #vy=diff(y)
  #x=array(x[0:-1])
  #y=array(y[0:-1])
  #bergLocVel.append([x,y,vx,vy])

#X=[]
#Y=[]
#VX=[]
#VY=[]
#for i in range(0,size(bergLocVel,0)):
  #X.extend(bergLocVel[i][0])
  #Y.extend(bergLocVel[i][1])
  #VX.extend(bergLocVel[i][2])
  #VY.extend(bergLocVel[i][3])

#X=array(X)
#Y=array(Y)
#VX=array(VX)
#VY=array(VY)

#nz=(where(VX!=0.0) or where(VY!=0.0))[0]

#X=X[nz]
#Y=Y[nz]
#VX=VX[nz]
#VY=VY[nz]


#nxcells=20
#nycells=20

#grid_x, grid_y = np.mgrid[X.min():X.max():complex(nxcells), Y.min():Y.max():complex(nycells)]

#vx_rbf=Rbf(X,Y,VX,function='linear')
#vxint=vx_rbf(grid_x.flatten(),grid_y.flatten())
#vxint=vxint.reshape([nxcells,nycells])
#vy_rbf=Rbf(X,Y,VY,function='linear')
#vyint=vy_rbf(grid_x.flatten(),grid_y.flatten())
#vyint=vyint.reshape([nxcells,nycells])

##subplot(1,2,1)
##imshow(vxint,extent=[grid_x[:,0].min(),grid_x[:,0].max(),grid_y[0].min(),grid_y[0].max()])
##colorbar()
##subplot(1,2,2)
##imshow(vyint,extent=[grid_x[:,0].min(),grid_x[:,0].max(),grid_y[0].min(),grid_y[0].max()])
##colorbar()
##show()

#figure()
#hold(True)

##imshow(sqrt(vxint**2+vyint**2),extent=[grid_x[:,0].min(),grid_x[:,0].max(),grid_y[0].max(),grid_y[0].min()],interpolation='none')
##grid_y=grid_y[-1::-1,:]
##vyint=vyint[-1::-1,:]

#q=quiver(grid_x,grid_y,vxint,vyint,scale_units='xy', angles='xy', scale=1, color='c')
#arrowSize=10.0
#arrowSpeed=arrowSize*pixelVel
#arrowLabel='%d'%(arrowSpeed)+' cm/s'
#p=quiverkey(q,1.0*grid_x.max(),1.01*grid_y.max(),arrowSize,arrowLabel,coordinates='data',color='r')
##for k in range(0,shape(berglocs)[1]):
  ##pl=[bl[k] for bl in berglocs]
  ##y=[yx[0] for yx in pl]
  ##x=[yx[1] for yx in pl]
  ###check to see if the mode value is the last value (stationary or not found berg)
  ###if mode(y)[0]!=y[-5]:
  ###plot(x,y,'*')
  ##plot(x[0::samplingrate],y[0::samplingrate])
###axis([0,200,0,300])
#imshow(imread('2012mask.png'),origin='upper')

#show()





##ax=gca()
###ax.invert_yaxis()  
##show()  

##calculate the velocities
##velocity=[]






#figure()
#hold(True)
#for i in range(0,len(berglocs)):
  #y=berglocs[i][1][0]
  #print y
  #x=berglocs[i][1][1]
  #plot(x,y,'*')
#ax=gca()
#ax.invert_yaxis()  
#show()  


#berg1=[]
#for i in range(0,len(berglocs)):
  #berg1.extend([berglocs[i][0]])


#where(array(y1)!=1e99)[0]
#use old points
#find new locations and overwrite
#find unused new points
#apppend new points to new locations
    


  


#berg_locs[0][0:len(t0)]=t0 #do for first time only
##berg_locs[1][0:len(t1)]=t1

##for i in range(0,len(t0)):
  ##for j in range(0,len(t1)):
  #berg_locs=len(files)*[[[nan,nan]]*maxbergs] #generate blank matrix to infill data
    
  

