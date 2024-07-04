

class ModelHandler(object):
    def __init__(self):
        self.model = joblib.load('random_forest_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        self.feature_names = joblib.load('feature_names.joblib')

    def preprocess(self, input_data: dict) -> np.ndarray:
        try:
            input_data = pd.DataFrame(input_data, columns=self.feature_names)
            input_data = self.scaler.transform(input_data)
            return input_data
        except Exception as e:
            raise ValueError("Error preprocessing data: {}".format(str(e)))

    def predict(self, input_data: dict) -> np.ndarray:
        try:
            input_data = self.preprocess(input_data)
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            raise ValueError("Error making predictions: {}".format(str(e)))

def lambda_handler(event: dict, context: object) -> dict:
    try:
        if 'input_data' not in event:
            raise ValueError("Input data is missing")
        
        model_handler = ModelHandler()
        input_data = event['input_data']
        predictions = model_handler.predict(input_data)
        return {'predictions': predictions.tolist()}
    except Exception as e:
        raise ValueError("Error processing request: {}".format(str(e)))

