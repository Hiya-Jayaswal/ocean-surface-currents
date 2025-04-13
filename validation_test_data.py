'''

WORKING CODE FOR VALIDATION
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from pprint import pprint

'''SAVED WEIGHTS'''
X = np.load('/home/guest/Hiya/00model/npy_file/X.npy')


'''splitting training testing validation'''
X_train, X_temp= (train_test_split(X, test_size=0.2, random_state= 35)) 
X_val, X_test= (train_test_split(X_temp, test_size=0.5, random_state= 35)) 

'''loading trained model'''
model = load_model('/home/guest/Hiya/00model/cnn_lstm_89.h5')  


predicted_osc = model.predict(X_test) 
predicted_osc = predicted_osc.squeeze() 

'''plots'''
for i in range (len(X_test)):
    plt.figure(figsize=(12, 6))
    plt.title(f'Estimated Ocean Surface Currents {i}')
    plt.imshow(predicted_osc[i], cmap='viridis', origin='lower')  # Plot the first time step
    plt.colorbar(label='Estimated OSC')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    plt.savefig(f"/home/guest/Hiya/00model/plots/Estimated_osc_{i}")

'''plotting'''
np.save("estimated_osc.npy", predicted_osc)

'''comparison with actual'''
try:
    osc = np.load('/home/guest/Hiya/00model/npy_file/Y.npy') 
    # pprint(osc)# Load actual OSC data for comparison
    plt.figure(figsize=(12,6))
    
    plt.subplot(1, 2, 1)
    plt.title('Actual Ocean Surface Currents')
    plt.imshow(osc[1], cmap='viridis', origin='lower')  # Plot actual OSC
    plt.colorbar(label='Actual OSC')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Ocean Surface Currents ')
    plt.imshow(predicted_osc[1], cmap='viridis', origin='lower')  # Plot predicted OSC
    plt.colorbar(label='Predicted OSC') 
    
    plt.show()
except FileNotFoundError:
    print("Ground truth OSC data not available for comparison.")

