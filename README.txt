Main used python files description:

- Master2.ipynb:  Training for different models and a bit of testing
- Master2.py:     Contains only training part of Master2.ipynb 

- Trainers_ULM.py: Master2 trainers 

- Demo_test.ipynb: DBlink original code, was only used for generating training data for reproducibility



I did a reprodicibility test for DBlink using their own generated filament data, with 1000 samples with X_train.shape = [1000,300,1,32,32] and 250 val samples for validation.
The data was generated from Demo_test.ipynb and not trained using that file. The error u see in that file, I was trying with 128x128 img size. 
For context I msged the authors of DBlink which msged me they used 32x32 pixels instead of 128x128 in the original DBLink repository, so I changed scale factor from 4 to 1 to get 32x32.
Since 128x128 is about an order of magnitude longer training for each epoch with batch size 1.

For the reproducibility, I trained the model using the file Master2.py which has the corresponding hyperparameters (,I increased the batch size to 16 for speed and epoch is set to 30, which has a training time about 3 days ). The trained model parameter weights is called DBlinkBase_model.
The epoch_loss.png contains the validation training loss during the training. 
To make a test result video, the Demo_test.ipynb can be used, I changed the sum_factor in the test section because the generated data already has sum_factor of 10 applied.

So far I did not reproduce the model weights in the original DBLink repository, since it seems the original model weights was trained on 128x128 imgs (as seen on expected output vids), contradicting the msg I got from the authors.
The trained model also did not perform well on testing (visually from the test video and test mse wise) for the generated 32x32 imgs (process the video result using trained model).

Update V1.1: 
Updated Master2.ipynb so that it can be used to both train and test DBLink for reproducibility testing and training (Only 1 cell needs to be skipped for testing DBLINK)

Added BaseX_test and Basey_test for testing the DBLink model I trained using 32x32 img size as mentioned above.

