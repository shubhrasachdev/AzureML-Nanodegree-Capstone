import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({"Pregnancies": pd.Series([0], dtype="int64"), 
                             "Glucose": pd.Series([0], dtype="int64"), 
                             "BloodPressure": pd.Series([0], dtype="int64"),
                             "SkinThickness": pd.Series([0], dtype="int64"), 
                             "Insulin": pd.Series([0], dtype="int64"),
                             "BMI": pd.Series([0.0], dtype="float64"), 
                             "DiabetesPedigreeFunction": pd.Series([0.0], dtype="float64"), 
                             "Age": pd.Series([0], dtype="int64")
                            })
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    #model_path = Model.get_model_path("best_hyperdrive_model")
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
