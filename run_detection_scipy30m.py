from numpy import *
from matplotlib.pyplot import *
import glob
import os
from scipy.stats import mode
from scipy import ndimage
from datetime import datetime, date, time, timedelta
from scipy.ndimage.measurements import label as cclabel
import time

def prepare(filename):
  #read annd preprocess each file by blurring,thersholding,blurring.
  img=imread(filename)
  img=img*mask
  img=ndimage.gaussian_filter(img, sigma1) #GAUSSIAN FILTER STEP 1
  img=where(img>thresh1*img.max(),1.0,0.0) #THRESHOLD STEP 1
  img=ndimage.gaussian_filter(img, sigma2) #GAUSSIAN FILTER STEP 2
  img=img[ymin:ymax,xmin:xmax]
  return img

def detect(filename):
  img=prepare(filename)
  #make assumption that edge pixels are all zeros
  img[:,0]=0
  img[:,-1]=0
  img[0,:]=0
  img[-1,:]=0
  coeff=thresh2#THRESHOLD STEP 2

  Y,X=where(img>coeff*img.max())
  img=zeros(shape(img))
  img[Y,X]=1
  #use scipy's built-in blob detection to get berg/blob locations and how many there are
  #generate 8-connected structure
  s = [[1,1,1],[1,1,1],[1,1,1]]
  blobs_new,nblobs=cclabel(img,structure=s)
      
  #find centroids of each berg (may also be able to use scipy.ndimage.measurements.center_of_mass)
  centroids=[]
  for i in range(1,int(blobs_new.max())+1):
    yv,xv=where(blobs_new==i)
    #coordinates=[ymin+(ymax-mean(yv)),xmin+mean(xv)]
    coordinates=[ymin+mean(yv),xmin+mean(xv)]
    centroids.append(coordinates)
  return centroids

def track_filelist(files):
  #figure out filename for output file using start and end images
  startFile=os.path.basename(files[0])[0:15]
  endFile=os.path.basename(files[-1])[0:15]
  outputfilename=startFile+'_'+endFile
  
  icelocs=[]
  for filenm in files:
    print filenm
    icedata=detect(filenm)
    icelocs.extend([icedata]) #icelocs contains all of the centroids for each image
    
  #maxdist=3.0 #specify maximum pixel distance between images to find nearest neighbor

  berglocs=[]
  berglocs.extend([icelocs[0]]) # this initializes the first starting icebergs

  for n in range(1,len(icelocs)): #this iterates over the data in a number of images
    #get distances for hypotenuse
    y0=[]
    x0=[]
    for yx in berglocs[n-1]: y0.append(yx[0]);x0.append(yx[1])
    y1=[]
    x1=[]
    for yx in icelocs[n]: y1.append(yx[0]);x1.append(yx[1])
    
    #find closest bergs to n-1 image
    newlocs=list(berglocs[n-1])
    idx_list=[]
    for i in range(0,len(berglocs[n-1])):
	dist=hypot(y1-y0[i],x1-x0[i])
	if size(dist)!=0:
	  idx=argmin(dist)
	  if dist[idx]<maxdist:
	    newlocs[i]=[y1[idx],x1[idx]]
	    idx_list.append(idx)
	    y1[idx]=1e99
	    x1[idx]=1e99
    berglocs.extend([newlocs])

  save('recscipy/'+outputfilename,berglocs)
  return

#######################################################################################
#specify all variables here
sigma1=1.0
sigma2=1.0
thresh1=0.3
thresh2=0.3
maxdist=2.0 
maskname='2012maskT.png'
directory='img/'
filetype='*.png'


mask=imread(maskname)
mask=mask.astype('bool')
mask=mask.astype('float')
yvl,xvl=where(mask==1.0) #find mask boundaries
ymin=yvl.min()
ymax=yvl.max()
xmin=xvl.min()
xmax=xvl.max()

#need to read in all files, separate them by 1hr intervals to produce hourly current maps. Have the end file of the last file be the first file in the next one
files=glob.glob(os.path.join(directory+filetype))
files.sort()

#make list of datetimes corresponding to filenames
measured_times=[datetime.strptime(os.path.basename(onefile)[0:15],"%Y%m%d_%H%M%S") for onefile in files]

#time difference for acquisitions (minutes)
timeDiff=30. 
timeCounter=0
timeList=[0]
#this code gets indices of hourly values so we can extract range of the file list and process on an hour-by-hour basis
for i in range(len(files)):
  #deltat is the time difference in minutes between two images
  deltat=((measured_times[i]-measured_times[0]).total_seconds())/60.
  if deltat >= timeDiff*(timeCounter+1): #changed >= to > to account for full hour
    timeCounter=timeCounter+1
    timeList.append(i)
    
for i in range(0,len(timeList)-1):
  print 'running tracking for: '+str(files[timeList[i]])+' to '+str(files[timeList[i+1]])
  track_filelist(files[timeList[i]:timeList[i+1]+1])#add 1 to the last value to get full hour of data
  
  
  