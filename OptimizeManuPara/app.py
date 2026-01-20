import pandas as pd
import numpy as np
np.random.seed(42)
data=pd.DataFrame({
    'machine_speed': np.random.uniform(50, 150, 500),
    'temperature': np.random.uniform(200, 400, 500),
    'pressure': np.random.uniform(30, 100, 500),
    'raw_material_quality': np.random.uniform(1, 10, 500),
    'operator_experience': np.random.randint(1, 15, 500),
})
data['defect_rate'] = (
    0.02 * data['machine_speed']
    + 0.03 * data['temperature']
    - 0.04 * data['pressure']
    - 0.5 * data['raw_material_quality']
    - 0.3 * data['operator_experience']
    + np.random.normal(0, 5, 500)
)
print(data.head())

from pycaret.regression import *

regression_setup=setup(
    data=data,
    target='defect_rate',
    session_id=123,
    normalize=True,
    train_size=0.8
)
best_model=compare_models()
tune_model=tune_model(best_model)
evaluate_model(tune_model)
ensemble = ensemble_model(best_model)
final_model=finalize_model(tune_model)
predictions = predict_model(final_model, data=data)
print(predictions.head())

