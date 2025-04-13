
# cnn_LSTM_train_test
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, r2_score
import seaborn as sns
from tensorflow.keras.utils import plot_model
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

'''loading the data'''
sst_file = '/home/guest/Hiya/nan_data/sst_169_nan.nc'
sss_file = '/home/guest/Hiya/nan_data/nansss.nc'
sla_file = '/home/guest/Hiya/nan_data/nansla.nc'
osc_file = '/home/guest/Hiya/nan_data/nanosc.nc'
wind_stress_file = '/home/guest/Hiya/nan_data/remap_wind.nc'

'''interpolate the sss data for daily data'''
def interpolate_temporal_sss(sss_data):

    ds = sss_data.to_dataset()
   

    start_date = ds.time.values[0]
    end_date = ds.time.values[-1]
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
   
    # Interpolate to daily values
    ds_interp = ds.interp(time=daily_dates, method='linear')
   

    return ds_interp['SSS']

def load_data(file_path, variable_name, temporal_interpolation=False):
    
    dataset = xr.open_dataset(file_path)
    data = dataset[variable_name]
    if temporal_interpolation and variable_name == 'SSS':
        data = interpolate_temporal_sss(data)
    return data


'''variables of the dataset with the interpolation in sss'''
sst_data = load_data(sst_file, 'sst')
sss_data = load_data(sss_file, 'SSS', temporal_interpolation=True)  # Apply temporal interpolation
sla_data = load_data(sla_file, 'sla')
u_data = load_data(osc_file, 'U')
v_data = load_data(osc_file, 'V')
u_wind_data = load_data(wind_stress_file, 'U')  
v_wind_data = load_data(wind_stress_file, 'V') 
# Renaming time dimension for consistency
u_data = u_data.rename({'Time':'time'})
v_data = v_data.rename({'Time':'time'})
lat = load_data(sst_file, 'Latitude').values
lon = load_data(sst_file, 'Longitude').values

'''common dates'''
def get_common_dates(*datasets):
    date_sets = [set(ds.time.values) for ds in datasets]
    common_dates = sorted(list(set.intersection(*date_sets)))
    return common_dates


'''
filter data to work on the common dates only
'''
common_dates = get_common_dates(sst_data, sss_data, sla_data, u_data, v_data, u_wind_data, v_wind_data)
sst = sst_data.sel(time=common_dates).values
sss = sss_data.sel(time=common_dates).values
sla = sla_data.sel(time=common_dates).values
u = u_data.sel(time=common_dates).values
v = v_data.sel(time=common_dates).values
''' squeeze the dimensions to reduce array size'''
u_wind = u_wind_data.sel(time=common_dates).values
u_wind  = np.squeeze(u_wind, axis = 1)
v_wind = v_wind_data.sel(time=common_dates).values
v_wind  = np.squeeze(v_wind, axis = 1)



def interpolate_missing_values(data):
    """ Interpolate missing values (NaNs) in 2D spatial data. """
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    mask = ~np.isnan(data)
    return griddata((x[mask], y[mask]), data[mask], (x, y), method='linear', fill_value=np.nanmean(data))




for i in range(sst.shape[0]):
    sst[i] = interpolate_missing_values(sst[i])
    sss[i] = interpolate_missing_values(sss[i])
    sla[i] = interpolate_missing_values(sla[i])
    u_wind[i] = interpolate_missing_values(u_wind[i])
    v_wind[i] = interpolate_missing_values(v_wind[i])
    
'''magnitude'''
osc = np.sqrt(u**2 + v**2)
osc  = np.squeeze(osc, axis = 1)
wind_stress = np.sqrt(u_wind**2 + v_wind**2)

'''normalisation(min-max)'''
sst = (sst - np.nanmin(sst)) / (np.nanmax(sst) - np.nanmin(sst))
sss = (sss - np.nanmin(sss)) / (np.nanmax(sss) - np.nanmin(sss))
sla = (sla - np.nanmin(sla)) / (np.nanmax(sla) - np.nanmin(sla))
osc = (osc - np.nanmin(osc)) / (np.nanmax(osc) - np.nanmin(osc))
wind_stress = (wind_stress - np.nanmin(wind_stress)) / (np.nanmax(wind_stress) - np.nanmin(wind_stress))


'''stacking the data and reshaping the osc size'''
X = np.stack((sst, sss, sla,wind_stress), axis=-1)
Y = osc.reshape(osc.shape[0], 81, 89, 1)

'''npy save to be used for testing '''
np.save("/home/guest/Hiya/00model/npy_file/X.npy", X)
np.save("/home/guest/Hiya/00model/npy_file/Y.npy", Y)

'''dividing the data in training data, testing and validation'''
X_train, X_temp, Y_train, Y_temp = (train_test_split(X,Y, test_size=0.2, random_state= 35)) 
X_val, X_test, Y_val, Y_test = (train_test_split(X_temp,Y_temp, test_size=0.5, random_state= 35))


print(f"Training set: {X_train.shape}, {Y_train.shape}")
print(f"Validation set: {X_val.shape}, {Y_val.shape}")
print(f"Testing set: {X_test.shape}, {Y_test.shape}")
# learning_rate = 0.001
'''Using CNN and LSTM model, using optimizer adam, and activation fn relu'''
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(81, 89, 4)),
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Reshape((1, -1)), 
    LSTM(64, activation='relu', return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(81 * 89, activation='linear'),
    Reshape((81, 89, 1))
])

model.compile(optimizer = 'adam', loss='mean_squared_error')


print("\nVerifying temporal interpolation for SSS data:")
original_sss = load_data(sss_file, 'SSS', temporal_interpolation=False)
print("Original SSS time steps:", len(original_sss.time))
print("Interpolated SSS time steps:", len(sss_data.time))
print("\nSample of original dates:", original_sss.time.values[:5])
print("Sample of interpolated dates:", sss_data.time.values[:5])

'''running 125 epochs and batch size as 1'''
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=125, batch_size=1,

)

'''evaluation '''
test_loss = model.evaluate(X_test, Y_test)
print("Test Loss:", test_loss)


'''Plotting the Loss curve'''
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5, color='blue') 
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='orange')
plt.title('Training Loss', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.grid(True)  # To add grid lines
plt.show()
# Predictions
predicted_osc = model.predict(X)

# Visualize actual vs predicted currents
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Actual Ocean Surface Currents')
# plt.imshow(Y[0].squeeze(), cmap='viridis', origin='lower')
plt.contourf(lon, lat, Y[0].squeeze(), cmap='viridis', levels=np.linspace(np.nanmin(predicted_osc), np.nanmax(predicted_osc), 100))

plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Predicted Ocean Surface Currents')
# plt.imshow(predicted_osc[0].squeeze(), cmap='viridis', origin='lower')
plt.contourf(lon, lat, predicted_osc[0].squeeze(), cmap='viridis', levels=np.linspace(np.nanmin(predicted_osc), np.nanmax(predicted_osc), 100))

plt.colorbar()
plt.show()

# # Flatten the actual and predicted data for confusion matrix
# Y_flat = Y.flatten()
# predicted_flat = predicted_osc.flatten()

# # Define bins for categorizing continuous data
# num_bins = 5
# bins = np.linspace(0, 1, num_bins + 1)

# def categorize_data(data, bins):
#     """Convert continuous data to categorical bins."""
#     labels = np.digitize(data, bins) - 1
#     return labels

# # Categorize actual and predicted values
# actual_labels = categorize_data(Y_flat, bins)
# predicted_labels = categorize_data(predicted_flat, bins)

# # Compute confusion matrix
# conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
#             xticklabels=np.arange(num_bins), yticklabels=np.arange(num_bins))
# plt.xlabel('Predicted Labels')
# plt.ylabel('Actual Labels')
# plt.title('Confusion Matrix')
# plt.show()

# # Generate model flowchart
# plot_model(model, to_file='model_flowchart.png', show_shapes=True, show_layer_names=True)

'''Evaluation metrics using MSE,RMSE,MAE'''
mse = mean_squared_error(Y.flatten(), predicted_osc.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y.flatten(), predicted_osc.flatten())
ss_res = np.sum((Y.flatten() - predicted_osc.flatten())**2)
ss_tot = np.sum((Y.flatten() - np.mean(Y.flatten()))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")
print("\nModel Performance Metrics:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()
# # Predictions
# predicted_osc = model.predict(X_test)

# # Visualize actual vs predicted currents
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.title('Actual Ocean Surface Currents')
# plt.imshow(Y_test[0].squeeze(), cmap='viridis', origin='lower')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.title('Predicted Ocean Surface Currents')
# plt.imshow(predicted_osc[0].squeeze(), cmap='viridis', origin='lower')
# plt.colorbar()
# plt.show()



# # Evaluation metrics
# mse = mean_squared_error(Y_test.flatten(), predicted_osc.flatten())
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(Y_test.flatten(), predicted_osc.flatten())
# ss_res = np.sum((Y_test.flatten() - predicted_osc.flatten())**2)
# ss_tot = np.sum((Y_test.flatten() - np.mean(Y.flatten()))**2)
# r_squared = 1 - (ss_res / ss_tot)

# print(f"R-squared: {r_squared:.4f}")
# print("\nModel Performance Metrics:")
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("MAE:", mae)

'''save to h5'''
model.save("/home/guest/Hiya/data_2023/cnn_lstm_89.h5")








