import pandas as pd
import numpy as np
from pycaret.anomaly import *
np.random.seed(42)

factory_data=pd.DataFrame({
    "temperature":np.random.normal(60,5,300),
    "pressure":np.random.normal(30,3,300),
    "vibration":np.random.normal(10,1,300)
})
anomaly_setup=setup(
    data=factory_data,
    normalize=True,
    session_id=123
)
iforest_model=create_model("iforest")
anomaly_results=assign_model(iforest_model)
print(anomaly_results.head())
save_model(iforest_model,"factory_anomaly_model")
print("\nModel saved successfully!")
loaded_model=load_model("factory_anomaly_model")
print("\nModel loaded successfully!")
new_sensor_data=pd.DataFrame({
    "temperature":[62,120],
    "pressure":[25,85],
    "vibration":[9,30]
})
prediction_results=predict_model(loaded_model,new_sensor_data)
print(prediction_results)

for index,row in prediction_results.iterrows():
    if row["Anomaly"]==1:
        print(f"\n⚠️ ALERT: Sensor reading {index} is ANOMALOUS. Action required!")
    else:
        print(f"\n✅ Sensor reading {index} is NORMAL. System OK.")