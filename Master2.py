# imports
import os
import torch
from DataHandlers import *
#from nn_models.NN_model_BN import *
from nn_models.lstm_unet import UNet_ConvLSTM
from nn_models.DBlink_NN import *

from Trainers_ULM import *
from Utils import *
from demo_exp_params import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torchvision
from utils.utilities import *
# from torchview import draw_graph

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
path = r'./' # Path to model

tmp_result_dir_exist = os.path.exists("./tmp_results")
if not tmp_result_dir_exist:
   # Create a tmp_results dir because it does not exist
   os.makedirs("./tmp_results")
    


method = 'DBlinkBase'  # options : DeepSMV / DBlink / Reg_Unet / DBlinkBase

TrainNetFlag = True

if(TrainNetFlag):
    X_train = torch.load('X_train')
    y_train = torch.load('y_train')
    X_val = torch.load('X_val')
    y_val = torch.load('y_val')

if method == 'DBlink':
    
    # model parameters
    model_name = 'DBlink_model'
    img_size = 128 
    num_layers = 2 # The number of LSTM layers
    hidden_channels = 4 # The hidden layer number of channels at the output of each lstm cell. Purpose?: adding more combinations of features? -> Higher complexity. It's more features.
    window_size = 12 # The number of used windows (in each direction) for the inference of each reconstructed frame

    model = ConvOverlapBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)  #DBlink
    
    if(TrainNetFlag):
        
        #training parameters 
        lr = 1e-3 # Training learning rate #was 1e-4
        betas = (0.99, 0.999) # Parameters of Adam optimizer
        epochs = 40
        batch_size =4
        patience = 5 # was 8
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)
        
        # loading data 
        
        y_train = y_train[:,[4],:]
        y_val = y_val[:,[4],:]
        
        dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
        dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

        trainer = DBlink_trainer(model, criterion, optimizer, scheduler, batch_size, window_size=window_size,
                                   vid_length=X_train.shape[1], patience=patience, device=device, modelname = model_name)
        trainer.fit(dl_train, dl_val, num_epochs=epochs)
        #torch.save(model.state_dict(), model_name)
        
    else:  # Testing
        model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
        

#------------------------------------------------------------------------------------------
        
    

elif method == 'DBlinkBase':
    
    # model parameters
    model_name = 'DBlinkBase_model'
    img_size = 32
    num_layers = 2 # The number of LSTM layers
    hidden_channels = 4 # The hidden layer number of channels at the output of each lstm cell. Purpose?: adding more combinations of features? -> Higher complexity. It's more features.
    window_size = 25 # The number of used windows (in each direction) for the inference of each reconstructed frame

    model = ConvOverlapBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)  #DBlink
    
    if(TrainNetFlag):
        
        #training parameters 
        lr = 1e-4 # Training learning rate #was 1e-4
        betas = (0.99, 0.999) # Parameters of Adam optimizer
        epochs = 25
        batch_size = 16
        patience = 3 # was 8
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)
        
        # loading data 
        if(TrainNetFlag):
            
            X_train = torch.load('BaseX_train')
            y_train = torch.load('Basey_train')
            X_val = torch.load('BaseX_val')
            y_val = torch.load('Basey_val')
        
        #y_train = y_train[:,[4],:]
        #y_val = y_val[:,[4],:]
        
        dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
        dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

        trainer = LSTM_overlap_Trainer(model, criterion, optimizer, scheduler, batch_size, window_size=window_size,
                                   vid_length=X_train.shape[1], patience=patience, device=device, modelname = model_name)
        trainer.fit(dl_train, dl_val, num_epochs=epochs)
        #torch.save(model.state_dict(), model_name)
        
    else:  # Testing
        model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    
    
 #----------------------------------------------------------------------------------------------------------   
    
elif method == "DeepSMV":
    
    # model parameters
    model_name = 'DeepSMV_model' 
    num_lstm_layers = 1 
    in_ch = 1  
    out_ch = 1 

    #model = UNet_ConvLSTM(n_channels=1, n_classes=1, use_LSTM=True, parallel_encoder=False, lstm_layers=1).to(device)  #DeepSMV original 
    model = UNet_ConvLSTM(n_channels= in_ch, n_classes= out_ch, use_LSTM=True, parallel_encoder=False, lstm_layers= num_lstm_layers).to(device)  #DeepSMV original 
    
    if(TrainNetFlag):
        
        #training parameters 
        lr = 1e-3  
        betas = (0.99, 0.999) 
        epochs = 40
        batch_size =8
        patience = 3 # was 8
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)
        
        # loading data 
        y_train = y_train[:,[4],:]  #velocitymap
        y_val = y_val[:,[4],:]      #velocitymap
        
        dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
        dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

        trainer = Deepsmv_trainer(model, criterion, optimizer, scheduler, batch_size,
                                   vid_length=X_train.shape[1], patience=patience, device=device ,modelname = model_name)
        trainer.fit(dl_train, dl_val, num_epochs=epochs)
        #torch.save(model.state_dict(), model_name)
        
    else:  # Testing
        model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
        
        
#---------------------------------------------------------------------------------------------------------------------------------------
    
elif method == 'Reg_Unet':
    
    # model parameters
    model_name = 'Reg_Unet_model' 
    num_lstm_layers = 1 
    in_ch = 2
    out_ch = 1 
    LSTM_used = False

    #model = UNet_ConvLSTM(n_channels=1, n_classes=1, use_LSTM=True, parallel_encoder=False, lstm_layers=1).to(device)  #DeepSMV original 
    model = UNet_ConvLSTM(n_channels= in_ch, n_classes= out_ch, use_LSTM= LSTM_used, parallel_encoder=False, lstm_layers= num_lstm_layers).to(device)  #DeepSMV original 
    
    if(TrainNetFlag):
        
        #training parameters 
        lr = 1e-3 
        betas = (0.99, 0.999) 
        epochs = 100
        batch_size = 4
        patience = 5 # was 8
        #criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)
        
        # loading data 
        X_train = y_train[:,[3,1],:]
        X_val = y_val[:,[3,1],:]
        
        y_train = y_train[:,[4],:]  #velocitymap
        y_val = y_val[:,[4],:]      #velocitymap
        
        dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
        dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

        trainer = Deepsmv_trainer(model, criterion, optimizer, scheduler, batch_size,
                                   vid_length=X_train.shape[1], patience=patience, device=device,modelname = model_name)
        trainer.fit(dl_train, dl_val, num_epochs=epochs )
        #torch.save(model.state_dict(), model_name)
        
    else:  # Testing
        model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
    
else:
    print('check model name')