   
    
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

''' fn to load model'''

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

'''for loading netcdf file'''
def load_netcdf_data(file_path, variable_name):
    dataset = xr.open_dataset(file_path)
    return dataset[variable_name]
'''interpolating nan values in the grid'''
def interpolate_missing_values(data):
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    mask = ~np.isnan(data)
    return griddata(
        (x[mask], y[mask]),
        data[mask],
        (x, y),
        method='linear',
        fill_value=np.nanmean(data)
    )

'''preprocessing (interpolate, normalisation, stacking)'''
def preprocess_data(sst, sss, sla, wind_stress):
    sst = interpolate_missing_values(sst)
    sss = interpolate_missing_values(sss)
    sla = interpolate_missing_values(sla)
    wind_stress = interpolate_missing_values(wind_stress)

 
    sst = (sst - np.nanmin(sst)) / (np.nanmax(sst) - np.nanmin(sst))
    sss = (sss - np.nanmin(sss)) / (np.nanmax(sss) - np.nanmin(sss))
    sla = (sla - np.nanmin(sla)) / (np.nanmax(sla) - np.nanmin(sla))
    wind_stress = (wind_stress - np.nanmin(wind_stress)) / (np.nanmax(wind_stress) - np.nanmin(wind_stress))

    # Stack
    X = np.stack((sst, sss, sla, wind_stress), axis=-1)
    X = X[np.newaxis, ...] 
    return X


'''for model estimation'''
def predict_ocean_surface_currents(model, sst, sss, sla, wind_stress):
    X_preprocessed = preprocess_data(sst, sss, sla, wind_stress)
    predicted_osc = model.predict(X_preprocessed)
    return predicted_osc.squeeze()

'''loading variables'''
def prepare_data_for_prediction(sst_file, sss_file, sla_file, wind_stress_file):
    sst_data = load_netcdf_data(sst_file, 'sst')
    sss_data = load_netcdf_data(sss_file, 'so')
    sla_data = load_netcdf_data(sla_file, 'sla')
    u_wind_data = load_netcdf_data(wind_stress_file, 'U')
    v_wind_data = load_netcdf_data(wind_stress_file, 'V')
    lat = load_netcdf_data(sst_file, 'Latitude')
    lon = load_netcdf_data(sst_file, 'Longitude')
    

    date = str(sst_data.time.values[0])

    sst = sst_data.sel(time=date, method='nearest').squeeze().values
    sss = sss_data.sel(time=date, method='nearest').squeeze().values
    sla = sla_data.sel(time=date, method='nearest').squeeze().values
    u_wind = u_wind_data.sel(time=date, method='nearest').squeeze().values
    v_wind = v_wind_data.sel(time=date, method='nearest').squeeze().values

    wind_stress = np.sqrt(u_wind**2 + v_wind**2)

    return sst, sss, sla, wind_stress, lat, lon

'''plotting '''
def visualize_prediction(lat, lon, osc, predicted_osc):
    # Define the color range for the plots
    min_val = min(np.nanmin(osc), np.nanmin(predicted_osc))
    max_val = max(np.nanmax(osc), np.nanmax(predicted_osc))

    plt.figure(figsize=(16, 6))

    # Plot Actual Ocean Surface Currents
    plt.subplot(1, 2, 1)
    plt.title('Actual Ocean Surface Currents')
    osc_masked = np.ma.masked_invalid(osc[-1].squeeze())  # Mask NaN values
    contour_actual = plt.contourf(
        lon, lat, osc_masked, cmap='Blues', levels=np.linspace(min_val, max_val, 500))
    plt.colorbar(contour_actual, label='Actual OSC')

    # Plot Predicted Ocean Surface Currents
    plt.subplot(1, 2, 2)
    plt.title('Predicted Ocean Surface Currents')
    predicted_osc_masked = np.ma.masked_invalid(predicted_osc.squeeze())  # Mask NaN values
    contour_predicted = plt.contourf(
        lon, lat, predicted_osc_masked, cmap='Blues', levels=np.linspace(min_val, max_val, 500)
    )
    plt.colorbar(contour_predicted, label='Predicted OSC')

    # # Overlay land areas (NaN regions) with white
    # land_mask = np.isnan(predicted_osc.squeeze())
    # plt.contourf(lon, lat, land_mask, colors='white')

    # Display the plots
    plt.show()
'''main'''

def main():
    model_path = "/home/guest/Hiya/00model/cnn_lstm_89.h5"
    sst_file = '/home/guest/Hiya/DATA_Neerajsir/data/13febsst00.nc'
    sss_file = '/home/guest/Hiya/DATA_Neerajsir/data/13febsss00.nc'
    sla_file = '/home/guest/Hiya/DATA_Neerajsir/data/13febsla00.nc'
    wind_stress_file = '/home/guest/Hiya/DATA_Neerajsir/data/13febwind00.nc'

    model = load_trained_model(model_path)

    osc = np.load("/home/guest/Hiya/DATA_Neerajsir/data/nanosc_13feb.npy")

    # plt.figure(figsize=(8, 6))
    # plt.title('Actual Ocean Surface Currents')
    # plt.imshow(osc[-1], cmap='viridis', origin='lower')
    # plt.colorbar(label='Actual OSC')

    try:
        sst, sss, sla, wind_stress, lat, lon = prepare_data_for_prediction(
            sst_file, sss_file, sla_file, wind_stress_file
        )

        predicted_osc = predict_ocean_surface_currents(model, sst, sss, sla, wind_stress)

        visualize_prediction(lat, lon, osc, predicted_osc)
 

        print("Prediction completed successfully!")
        osc = np.nan_to_num(osc)
        #print(osc.dtype)
        '''Performance metrics using MSE,RMSE,R-squared'''
        mse = mean_squared_error(osc.flatten(), predicted_osc.flatten())
        rmse = np.sqrt(mse)
        print("\nModel Performance Metrics:")
        print("MSE:", mse)
        print("RMSE:", rmse)
        ss_res = np.sum((osc.flatten() - predicted_osc.flatten())**2)
        ss_tot = np.sum((osc.flatten() - np.mean(predicted_osc.flatten()))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"R-squared: {r_squared:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()