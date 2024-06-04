import numpy as np
import skimage as ski
import math
import matplotlib.pyplot as plt

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

def generate_simulated_photos(G,mu,imgsize,pixelsize,C,Ts,sumfactor,videolength,rho):
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
    print(N_MBs)
    
    N_MBs = round((h/2)**2 * np.pi * length * C     )
    print(N_MBs)

    
    img_mask = np.zeros((imgsize,imgsize),dtype= np.uint8)
    start =(h_pix, 0)   #(y,x)
    extent = (imgsize-h_pix,imgsize)
    rr,cc = ski.draw.rectangle(start, end=extent, shape=img_mask.shape)
    img_mask[rr,cc] = 255
    
    
    
    #mb simulation in one of the imgsize pixels.
    x_pix = np.random.randint(0,imgsize,N_MBs)
    y_pix = np.random.randint(h_pix, imgsize-h_pix,N_MBs)  # +1 since its [start,stop)

    
    y_maskpix = np.arange(h_pix , imgsize - h_pix , 1)
    u_mask = Poisseuille_pipe(G,mu,yy[y_maskpix]+(h/2),h)
    velocitymap = np.zeros((int(videolength/sumfactor),imgsize,imgsize),dtype=np.float32)
    real_u_mask = np.transpose(np.tile(u_mask,(imgsize,1)))
    velocitymap[:,y_maskpix[0]:y_maskpix[-1]+1,:] = real_u_mask
    
    #reynolds number:
    Re = Re_number(rho,(h/2),G,mu)
    print(Re)
    
    
    
    movementarray = np.zeros((2,N_MBs,videolength))  #making array for movement IRL positions
    movementarray[0,:,0] = yy[y_pix]
    movementarray[1,:,0] = xx[x_pix]
    
    movementarraypixel = np.zeros((2,N_MBs,videolength))  # making array for movement in pixel positions 
    movementarraypixel[:,:,0] = [y_pix,x_pix]
    
    #potentialxpos= np.zeros((N_MBs,1))

    
    for i in range(videolength-1):
        #update movement
        ds = Poisseuille_pipe(G,mu,movementarray[0,:,i]+h/2,h)*Ts
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


def generate_simulated_photos_real(G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel):
    # selecting vessel radius in the pixel range: [R_minpixel, R_maxpixel], vessel radius in pixels
   
    #R_maxpixel = 3.0
    
    if R_minpixel == R_maxpixel:
        R_pixel = R_minpixel 
    else:
        #R_choices = np.arange(R_minpixel,R_maxpixel+2)   
        R_choices = np.arange(R_minpixel,R_maxpixel+1) 
        R_pixel = np.random.choice(R_choices) # uniform distribution 
        
    if R_pixel < (R_maxpixel+1):
        
        
        R = R_pixel*pixelsize # vessel radius in m
        T_acq = N_frames*Ts # time sampling 
        
        u_mean = R**2 * G /(8*mu) # mean velocity in m/s in vessel with radius R
    
            
        #print(N_MBs) #sanity check
        
        #make distribution 
        parabolic_points =  np.linspace((R-pixelsize/2),-(R-pixelsize/2),int(R_pixel*2)) # 2*R_pixel number of points evenly spaced between [ -(R-pixelsize/2), (R-pixelsize/2)] 
        parabolic_speeds = Poisseuille_flow(G,mu,parabolic_points,R) 
        parabolic_pixel_speeds =  parabolic_speeds/ pixelsize # u(r) speeds in pixels/s
        p_chances= make_distribution(parabolic_speeds)
        
        #correcting MB concentration, since only a p_crossing chance of a MB passes through the vessel
        p_crossing = passing_probability(R,parabolic_points,p_chances) #p chance of crossing vessel of the sampled MB
        #print('p_crossing',p_crossing)
        correction = 1/p_crossing
        
        N_b = C *T_acq * u_mean * np.pi*R**2*correction  #expect # of MBs to be simulated for vessel with radius R_pixel in T_acq seconds
        #print('N_b real :',N_b/correction) 
        
        rng=np.random.default_rng() 
        
        # N_MBs to spawn since N_b is not an integer 
        if rng.random()< np.mod(N_b,1):  
            N_MBs= math.ceil(N_b)  
        else:
            N_MBs = math.floor(N_b) 
            
        #print('N_MBs spawned with correction :',N_MBs)
        
        
        # sampling mb spawn
        
        choice_ind =np.arange(R_pixel*2) # indices to choose from choice given parabolic distribution
        gap = (imgsize/2 - R_pixel ) # gap between side in y and R_pixel
        if gap>0:
            gap_range = np.arange(0,gap*2 + 1)
            gap = np.random.choice(gap_range)
    
        ss = []
        while len(ss)==0:
            
        
            samples_ind = np.random.choice(choice_ind, N_MBs , p = p_chances).astype(int) #sampling the ind choices
            
            y_locations = gap+samples_ind # y position of the imgsize x imgsize matrix # not ordered, pixel positions
            pixel_speeds = parabolic_pixel_speeds[samples_ind] #pixel speeds of the sampled indices
            
            
            XX = round(2*u_mean *T_acq/ pixelsize) # avg distance traveled for a mb with u mean speed during T_acq, in pixel distance
            x_sample_ind =  np.random.choice(XX,N_MBs)  # x position in pixel pos
            
            # position state of each MB starting from x_sample_ind, positions are in pixels !
            time_vector=np.array([ np.arange(N_frames)*Ts ])
            ds_pixel = np.matmul( np.transpose(np.array([pixel_speeds])) ,     time_vector) #movement in pixel ds  matrix size (N, Nframes)
            
            x_start = np.transpose( np.tile(x_sample_ind, ( int(N_frames),1)) )
            
            s = np.round(x_start + ds_pixel) # MB x pixel position states vs frames 
            
            #locating the time indices at which it passed through the ROI where the camera is. 
            window = [XX-imgsize , XX] #window in pixel x position 
            intersection_times =find_passing_times(s, window) # time of intersection for each object n x m, n is MB# and m is intersection time 
            #print(intersection_times)
            ss = intersection_times
        
        x_window = np.arange(window[0],window[1])
        x_locations = s[intersection_times[:,0],intersection_times[:,1]]
        X_ind = np.array([np.nonzero(x == x_window) for x in x_locations ])
        X_ind_real = X_ind[:,:,0]  # X positions
        
        indices = np.append(intersection_times,X_ind_real,axis = 1) # col 1 mb #, col2 time nframe , col3 x ind
        
        #counting the amount of MBs that passed
        N_passing_MBs = np.sum([ np.max(s[y,:]>=window[1]) for y in np.arange(N_MBs)])
        #print('real amount of MBs that passed through there :',N_passing_MBs)
        
        # Make localization array 
        observ = np.zeros((int(N_frames),imgsize,imgsize))
        observ[indices[:,1], y_locations[indices[:,0]].astype(int) , indices[:,2] ] = 1
        
        # Make velocity map groundtruth, speed in mm/s
        velocity = np.zeros((imgsize,1)) 
        velocity[int(gap):int(gap+R_pixel*2),0] = parabolic_speeds*1e3 
        velocitymap = np.tile(velocity,(1,imgsize))
        
        # Make vessel segment groundtruth 
        vessel_gt = velocitymap>0 
        vessel_gt = vessel_gt.astype(int)
        
        #print(N_passing_MBs/N_b)
        
        observ_stacked = np.zeros((int(N_frames/sumfactor),imgsize,imgsize )).astype(int)
        for i in range(int(N_frames/sumfactor)):
            observ_stacked[i,:]= np.sum(observ[i*sumfactor:(i+1)*sumfactor,:,:],axis = 0).astype(int)
        
    else:
        observ_stacked = np.zeros((int(N_frames/sumfactor),imgsize,imgsize )).astype(int)
        velocitymap = np.zeros((imgsize,imgsize))
        vessel_gt = np.zeros((imgsize,imgsize))
            
              
    return observ_stacked, velocitymap, vessel_gt

            

def make_distribution(u):
    summation = np.sum(u) #for normalization
    p = u/np.sum(u)
    return p  

def poisseuille_flow_speed(G,mu,y,h):
    u= G*y*(h-y)/(2*mu)
    # u = 0 if y = h or y= 0
    return u

def Poisseuille_pipe(G,mu,y,h):
    R = h/2 
    r = R-y 
    u = G*(R**2 - r**2)/(4*mu)
    return u

def Poisseuille_flow(G,mu,r,R): #poisseuile flow in pipe but uses r,R instead
    u = G*(R**2 - r**2)/(4*mu)
    Re = Re_number(1056,R,G,mu)
    #print(Re)
    
    return u

def Re_number(rho,R,G,mu):
    Re = rho*R**3 *G/(4*mu**2)
    return Re

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_passing_times(position_matrix, window_bounds):
    # Create boolean masks for positions within window bounds
    within_bounds = np.logical_and(position_matrix >= window_bounds[0], position_matrix < window_bounds[1])
 

    # Initialize a list to store passing times for each object
    passing_times_per_object = []

    # Iterate over each object
    for i, obj_passes in enumerate(within_bounds):
        # Get indices of passing times for the current object
        passing_times_indices = np.nonzero(obj_passes)[0]

        # Append passing times to the list if the object passes through the window
        if len(passing_times_indices) > 0:
            passing_times_per_object.extend([[i, t] for t in passing_times_indices])

    # Convert the list to a NumPy array
    passing_times_matrix = np.array(passing_times_per_object)

    return passing_times_matrix

def passing_probability(R,r,p_distribution): 
    #calculates the avg probability of passing through the vessel cross-section
    r_split = r[: int(r.size/2)]              
    r_prob= (1-r_split**2/R**2)
    p_prob = p_distribution[: int(r.size/2)]       #prob of spawning on x = r
    
    p_passing = np.dot(r_prob,p_prob)*2 
    #r is half the choices
    
    return p_passing

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

def create_dblink_data(N,imgsize,G,mu,pixelsize,C,Ts,sumfactor,videolength,rho):
    GT = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize))
    observation = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize)) 
    velocitymaps = np.zeros((N,int(videolength/sumfactor),imgsize,imgsize))
    for i in range(N):
        obs,groundtruth,_,_,velocity= generate_simulated_photos(G,mu,imgsize,pixelsize,C,Ts,sumfactor,videolength,rho)
        observation[i,:]= obs
        GT[i,:] = groundtruth
        velocitymaps[i,:] = velocity
        
    return observation[:,:,None,:,:], GT[:,:,None,:,:], velocitymaps[:,:,None,:,:]

def create_real_data(N,G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel):
    vessel_GT = np.zeros((N,imgsize,imgsize)) 
    observation_stacked = np.zeros((N,int(N_frames/sumfactor),imgsize,imgsize)) 
    velocitymap = np.zeros((N,imgsize,imgsize)) 
    
    for i in range(N):
        obs,vel,gt = generate_simulated_photos_real(G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel)
        
        observation_stacked[i,:]= obs
        vessel_GT[i,:] = gt
        velocitymap[i,:] = vel
  
    
    return observation_stacked[:,:,None,:,:], velocitymap , vessel_GT
    
    
def smallest_R(C,mu,G,T_acq,N_b,pixelsize):
    R4=(N_b*8*mu)/(C*T_acq*np.pi*G) # R^4
    R= R4**(1/4)
    pixels_smallest_radius = math.ceil(R/pixelsize) # ceilin of the smallest radius in pixel size for simulation
    return R, pixels_smallest_radius