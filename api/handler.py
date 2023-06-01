import pickle

import pandas as pd

from flask             import Flask, Response, request
from ChurnInd.ChurnInd import ChurnInd

# load model
model = pickle.load(open('/home/felipe/repos/churn_indi/src/model/def_model.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/ChurnInd/predict', methods=['POST'])
def ChurnInd_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else: # multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # instantiate Rossmann class
        pipeline = ChurnInd()
        
        # data cleaning
        #df1 = pipeline.data_cleaning(test_raw)
        
        # feature egineering
        df2 = pipeline.feature_engineering(test_raw)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
    return df_response

if __name__ == '__main__':
    app.run('0.0.0.0')
