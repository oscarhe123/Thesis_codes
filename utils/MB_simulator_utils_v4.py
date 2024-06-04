import numpy as np
import skimage as ski
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_simulated_photos_real(G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel):
    # selecting vessel radius in the pixel range: [R_minpixel, R_maxpixel], vessel radius in pixels
   
    #R_maxpixel = 3.0
    
    if R_minpixel == R_maxpixel:
        R_pixel = R_minpixel 
    else:
        R_choices = np.arange(R_minpixel,R_maxpixel+2)   
        R_pixel = np.random.choice(R_choices) # uniform distribution 
        
    if R_pixel < (R_maxpixel+1):
        
        
        R = R_pixel*pixelsize # vessel radius in m
        T_acq = N_frames*Ts # time sampling 
        
        G = G*np.random.randint(5,11,1)[0]*0.1 # variable G
        
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
        
        #C =  C*np.random.randint(5,11,1)*0.1
        #print(C)
        
        
        N_b = C *( T_acq *2* u_mean +imgsize*pixelsize)* np.pi*R**2  #expect # of MBs to be simulated for vessel with radius R_pixel in T_acq seconds
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
    
        ss = []  # a MB needs to pass through the section 
        while len(ss)==0:
            
        
            samples_ind = np.random.choice(choice_ind, N_MBs , p = p_chances).astype(int) #sampling the ind choices
            
            y_locations = gap+samples_ind # y position of the imgsize x imgsize matrix # not ordered, pixel positions
            pixel_speeds = parabolic_pixel_speeds[samples_ind] #pixel speeds of the sampled indices
            
            
            XX = round(2*u_mean *T_acq / pixelsize + imgsize) #  distance traveled for a mb with u mean speed during T_acq, in pixel distance
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
        
        # Make velocity tracks 
        velocity = np.zeros((imgsize,1)) 
        velocity[int(gap):int(gap+R_pixel*2),0] = parabolic_speeds*1e3 
        
        veltracks = np.zeros((imgsize,imgsize))
        parse_locations = {i: indices[indices[:, 0] == i] for i in np.unique(indices[:, 0])}
        densitytracks = np.zeros((len(parse_locations),imgsize,imgsize)) 
        k = 0 # counter
        
        for i in np.unique(indices[:, 0]):
            veltracks[y_locations[ parse_locations[i][0,0]].astype(int), parse_locations[i][0,-1]: (parse_locations[i][-1,-1] +1 ) ] = velocity[y_locations[ parse_locations[i][0,0]].astype(int) ]
            
            densitytracks[k, y_locations[ parse_locations[i][0,0]].astype(int), parse_locations[i][0,-1]:(parse_locations[i][-1,-1] +1 ) ] = 1
            k +=1
            
        tracks = np.zeros((imgsize,imgsize),dtype = int)
        tracks[veltracks!=0]= 1
        
        density_tracks = np.sum(densitytracks,axis=0) # density tracks
        
        
        # Make velocity map groundtruth, speed in mm/s
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
        veltracks = np.zeros((imgsize,imgsize))
        tracks = np.zeros((imgsize,imgsize))
        density_tracks = np.zeros((imgsize,imgsize))
        
        
            
              
    return observ_stacked, velocitymap, vessel_gt , veltracks, tracks ,density_tracks

            

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


def create_real_data(N,G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel):
    vessel_GT = np.zeros((N,imgsize,imgsize)) 
    observation_stacked = np.zeros((N,int(N_frames/sumfactor),imgsize,imgsize)) 
    velocitymap = np.zeros((N,imgsize,imgsize)) 

    veltracks = np.zeros((N,imgsize,imgsize)) 
    tracks= np.zeros((N,imgsize,imgsize)) 
    dentracks = np.zeros((N,imgsize,imgsize)) 
    
    for i in tqdm(range(N)):
        obs,vel,gt , veltra, tra , dentra = generate_simulated_photos_real(G,mu,imgsize,pixelsize,C,Ts,sumfactor,N_frames,R_minpixel,R_maxpixel)
        
        observation_stacked[i,:]= obs
        vessel_GT[i,:] = gt
        velocitymap[i,:] = vel
        
        veltracks[i,:] = veltra
        tracks[i,:] = tra
        dentracks[i,:] = dentra
      
    obs_summed = np.sum(observation_stacked ,axis = 1 )
    velocitymap = velocitymap[:,None,:,:]
    vessel_GT = vessel_GT[:,None,:,:]
    veltracks = veltracks[:,None,:,:]
    tracks = tracks[:,None,:,:]
    dentracks = dentracks[:,None,:,:]
    
    combined = obs_summed[:, None,:,:]                           # 0 
    combined = np.append(combined, tracks, axis = 1)             # 1
    combined = np.append(combined, vessel_GT, axis = 1)          # 2
    combined = np.append(combined, veltracks, axis = 1)          # 3
    combined = np.append(combined, velocitymap, axis = 1)        # 4
    combined = np.append(combined, dentracks, axis = 1)          # 5
    
    # 0 obs_summed
    # 1 tracks
    # 2 vessel GT
    
    # 3 vel tracks
    # 4 velocitymap
    # 5 density tracks
    

        
    return observation_stacked[:,:,None,:,:], combined


def smallest_R(C,mu,G,T_acq,N_b,pixelsize):
    correction = 2*0.8
    R4=(N_b*8*mu)/(C*T_acq*np.pi*G *correction) # R^4
    R= R4**(1/4)
    pixels_smallest_radius = round(R/pixelsize) # ceilin of the smallest radius in pixel size for simulation
    return R, pixels_smallest_radius


