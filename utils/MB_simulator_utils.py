import numpy as np
import skimage as ski

def generate_simulated_photos1(G,mu,imgsize,pixelsize,C,h,Ts,window):
    # make an img of summed localized MB frames and also its corresponding masks
    
    # G gradient pressure
    # mu dynamic visc
    # imgsize size of image in pixels
    # pixelsize resolution of each pixel
    # N number of simulated data
    # C concentration of MB
    # h distance between plates the two non-slip plates
    # Ts = sample time
    # window = window for summation 
    
    length =pixelsize*imgsize  # length and width of the img
    maxhight = length/2 
    yy = np.linspace(maxhight,-maxhight,imgsize)   # y to -y
    xx = np.linspace(0,length,imgsize)             # 0 to max x
    h_pix = np.random.randint(imgsize/8,imgsize/2- imgsize/16) # choosing max radius of plate.
    h = yy[h_pix]*2  #distance between two plates
    N_MBs = int(h*length*C)   #MBs in the img 
    
    #move pipe up or down, 
    move = np.random.randint(0,h_pix)  #how much distance to move
    move_choice = rand1()              #move in up or down direction
    move_updown = move*move_choice

    
    #make mask for the img
    img_mask = np.zeros((imgsize,imgsize),dtype= np.uint8)
    start =(h_pix + move_updown,0)   #(y,x)
    extent = (imgsize-h_pix + move_updown,imgsize)
    rr,cc = ski.draw.rectangle(start, end=extent, shape=img_mask.shape)
    img_mask[rr,cc] = 1

    
    #mb simulation in one of the imgsize pixels.
    x_pix = np.random.randint(0,imgsize,N_MBs)
    y_pix = np.random.randint(h_pix, imgsize-h_pix,N_MBs)
    # N_mask = imgsize -2*h_pix
    y_maskpix = np.arange(h_pix , imgsize - h_pix , 1)
    #print(y_pix.shape)
    
    u = poisseuille_flow_speed(G,mu,yy[y_pix]+(h/2),h)
    # print(u.min())
    y_pix = y_pix + move_updown
    move_s = u*Ts
    movementarray = np.zeros((N_MBs,window))
    movementarray[:,0] = xx[x_pix]
    
    #mask simulation for velocity
    u_mask = u = poisseuille_flow_speed(G,mu,yy[y_maskpix]+(h/2),h)
    y_maskpix = y_maskpix +move_updown
    # print(u_mask.shape)
    # print(N_mask)
    # print(y_maskpix.shape)
    # print(y_maskpix[0])
    movementarray_mask = np.zeros((imgsize , imgsize))
    real_u_mask = np.transpose(np.tile(u_mask,(imgsize,1)))
    movementarray_mask[y_maskpix[0]:y_maskpix[-1]+1,:] = real_u_mask
    # print(movementarray_mask.shape)
    
    
    #simulation of the individual mbs moving in the x-direction
    for i in range(N_MBs):
        k= 0 
        
        while k<(window-1) and movementarray[i,k]< length:
            k +=1 
            movementarray[i,k] = movementarray[i,k-1] + move_s[i]
        if k<(window-1) and movementarray[i,k]>= length:
            movementarray[i,k:] = movementarray[i,k-1]
        else:
            pass
        
      #simulation of the individual mbs moving in the x-direction for MASK      
        
        
    # finding the pixels where the MB is closest to in the simulation    
    movementarray_pixels = np.zeros((N_MBs,window))
    
    for i in range(N_MBs):
        for j in range(window):
            movementarray_pixels[i,j] = find_nearest(xx,movementarray[i,j])
    
    movementarray_pixels = movementarray_pixels.astype(int)
 
    
    # make image with summed localizations
    img_summedloc = np.zeros((imgsize,imgsize),dtype = np.uint8)

    locy = np.tile(y_pix, (window,1)).transpose().astype(int)

    
    
    for i in range(N_MBs):  
        img_summedloc[locy[i,:],movementarray_pixels[i,:]] = 1
    
    #make image with summed lines
    img_summedlines = np.zeros((imgsize,imgsize))
    img_summedlines_pix = np.zeros((imgsize,imgsize),dtype=np.uint8)
    distances = movementarray[:,1:] - movementarray[:,0:-1]  #distances moved in m
    distances_pix = movementarray_pixels[:, 1:] - movementarray_pixels[:,0:-1] #distances moved in pixels
    
    for i in range(N_MBs):
        for j in range(window-1):
            rr, cc = ski.draw.line(y_pix[i],movementarray_pixels[i,j] , y_pix[i], movementarray_pixels[i,j+1])
            img_summedlines[rr,cc] = distances[i,j]
            img_summedlines_pix[rr,cc] = distances_pix[i,j]
            
          
    
    # return img_mask,h_pix,extent,u,y_pix,movementarray,movementarray_pixels,img_summedloc,img_summedlines,img_summedlines_pix
    return img_mask,img_summedloc,img_summedlines,img_summedlines_pix,movementarray_mask

def generate_simulated_photos(G,mu,imgsize,pixelsize,C,Ts,sumfactor,videolength):
    # make an img of summed localized MB frames and also its corresponding masksB
    
    # G gradient pressure
    # mu dynamic visc
    # imgsize size of image in pixels
    # pixelsize resolution of each pixel
    # N number of simulated data
    # C concentration of MB
    # h distance between plates the two non-slip plates
    # Ts = sample time
    # window = window for summation 
    
    #Outputs
    #Observations = summed localization frame 
    #groundtruth = binary groundtruth of the vessel
    #
    
    length =pixelsize*imgsize  # length and width of the img IRL
    maxhight = length/2 
    yy = np.linspace(maxhight,-maxhight,imgsize)   # y to -y
    xx = np.linspace(0,length,imgsize)             # 0 to max x
    h_pix = np.random.randint(imgsize/8,imgsize/2- imgsize/16) # choosing max radius of plate.
    h = yy[h_pix]*2  #diameter
    N_MBs = int(h*length*C)   #MBs in the img 
    N_MBs = 10
    
    img_mask = np.zeros((imgsize,imgsize),dtype= np.uint8)
    start =(h_pix, 0)   #(y,x)
    extent = (imgsize-h_pix,imgsize)
    rr,cc = ski.draw.rectangle(start, end=extent, shape=img_mask.shape)
    img_mask[rr,cc] = 255
    
    
    
    #mb simulation in one of the imgsize pixels.
    x_pix = np.random.randint(0,imgsize,N_MBs)
    y_pix = np.random.randint(h_pix, imgsize-h_pix,N_MBs)  # +1 since its [start,stop)

    
    y_maskpix = np.arange(h_pix , imgsize - h_pix , 1)
    u_mask = poisseuille_flow_speed(G,mu,yy[y_maskpix]+(h/2),h)
    velocitymap = np.zeros((int(videolength/sumfactor),imgsize,imgsize),dtype=np.float32)
    real_u_mask = np.transpose(np.tile(u_mask,(imgsize,1)))
    velocitymap[:,y_maskpix[0]:y_maskpix[-1]+1,:] = real_u_mask
    
    
    
    movementarray = np.zeros((2,N_MBs,videolength))  #making array for movement IRL positions
    movementarray[0,:,0] = yy[y_pix]
    movementarray[1,:,0] = xx[x_pix]
    
    movementarraypixel = np.zeros((2,N_MBs,videolength))  # making array for movement in pixel positions 
    movementarraypixel[:,:,0] = [y_pix,x_pix]
    
    #potentialxpos= np.zeros((N_MBs,1))
    
    for i in range(videolength-1):
        #update movement
        ds = poisseuille_flow_speed(G,mu,movementarray[0,:,i]+h/2,h)*Ts
        movementarray[1,:,i+1] = movementarray[1,:,i] + ds
        movementarray[0,:,i+1] = movementarray[0,:,i] 
        xypos = np.transpose(movementarray[:,:,i+1])

        #check if xlimit is passed and replace y,x irl values then spawn new mbs
        newdata = [(yy[np.random.randint(h_pix, imgsize-h_pix)], xx[np.random.randint(0,imgsize)]) if x>= length else (y,x) for (y,x) in xypos]
        
        #video frames real location
        movementarray[:,:,i+1] = np.transpose(np.array(newdata))
        
        #video frames pixel locations
        pixeloc = [ (find_nearest(yy,y),find_nearest(xx,x)) for (y,x) in newdata]
        movementarraypixel[:,:,i+1] =  np.transpose(pixeloc)

        
    observations = np.zeros( (int(videolength/sumfactor),imgsize,imgsize ),dtype=np.uint8 ) # observation of summed frames, so videolength/sumfactor
    groundtruth = np.zeros( (int(videolength/sumfactor),imgsize,imgsize ) ,dtype=np.uint8 )  # ground truth vessel mask
    for i in range(int(videolength/sumfactor)):
        dummy = movementarraypixel[:,:,i*sumfactor:(i+1)*sumfactor].astype(np.uint8)  #all pixel points in the sumfactor frames
        dummy = np.reshape(dummy,(2,N_MBs*sumfactor))  # rearrange to (2,N_MBs*sumfactor) 

        dummy = np.transpose(dummy)
        ind = tuple(np.array(dummy).T)
        # print(ind)
        dummy2 = np.zeros((imgsize,imgsize))
        dummy2[ind]= 1 
        observations[i,:,:]= dummy2 
        groundtruth[i,:,:] = img_mask
        
    
    return observations,groundtruth,movementarraypixel,movementarray,velocitymap


def poisseuille_flow_speed(G,mu,y,h):
    u= G*y*(h-y)/(2*mu)
    # u = 0 if y = h or y= 0
    return u

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def rand1(): 
    return 1 if np.random.random() < 0.5 else -1

def create_data(N,imgsize,G,mu,pixelsize,C,h,Ts,window):
    img_masks = np.zeros([N,imgsize,imgsize],dtype=np.uint8)
    img_loc_sums = np.zeros([N,imgsize,imgsize],dtype=np.uint8)
    img_lines_dist = np.zeros([N,imgsize,imgsize])
    img_lines_dist_pix = np.zeros([N,imgsize,imgsize],dtype=np.uint8)
    movementarray_mask = np.zeros([N,imgsize,imgsize])
    
    for i in range(N):
        img_masks[i,:], img_loc_sums[i,:], img_lines_dist[i,:], img_lines_dist_pix[i,:], movementarray_mask[i,:]=generate_simulated_photos(G,mu,imgsize,pixelsize,C,h,Ts,window)
        print(i)
        
    return img_masks,img_loc_sums,img_lines_dist,img_lines_dist_pix, movementarray_mask
    # return img_masks,h_pix,extent,u,y_pix,movementarray,movementarray_pixels,img_summedloc,img_summedlines,img_summedlines_pix

def create_dblink_data(N,imgsize,G,mu,pixelsize,C,Ts,sumfactor,videolength):
    GT = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize))
    observation = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize)) 
    velocitymaps = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize))
    for i in range(N):
        obs,groundtruth,_,_,velocity= generate_simulated_photos(G,mu,imgsize,pixelsize,C,Ts,sumfactor,videolength)
        observation[i,:]= obs
        GT[i,:] = groundtruth
        velocitymaps[i,:] = velocity
        
    return observation[:,:,None,:,:], GT[:,:,None,:,:], velocitymaps[:,:,None,:,:]
    
