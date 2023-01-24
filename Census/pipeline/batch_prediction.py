from Census.exception import SensorException
from Census.logger import logging
from Census.predictor import ModelResolver
import pandas as pd
from Census.utils import load_object
import os,sys
from datetime import datetime
PREDICTION_DIR="prediction"
from sklearn.preprocessing import LabelEncoder

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace=True)
        #cleaning string in object columns
        for i in df.columns:
            if df[i].dtype=='object':
                df[i] = df[i].str.strip()

        #cleaning rows
        l=[]
        for i in df.columns:
            for j in range(df.shape[0]):
                if df[i][j]=='?':
                    l.append(j)
        df.drop(index=l,inplace=True)  

        #dropping some columns
        df.drop(['fnlwgt','capital-gain','capital-loss','education-num'],axis=1,inplace=True) 
        if 'salary' in df.columns:
            df.drop('salary',axis=1,inplace=True)

        # input featur encoding
        logging.info(f"Loading input_encoder to encode dataset")
        df_copy = df.copy()
   

        input_feature_encoder = load_object(file_path=model_resolver.get_latest_input_feature_encoder_path()) 

        #for test_df
        cat_col=[]
        for i in df_copy.columns:
            if df_copy[i].dtype=='object':
                cat_col.append(i)
        data_categorical = df_copy[cat_col]


        data_encoded = input_feature_encoder.transform(data_categorical)

        a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'country'])
        b = df[['age','hours-per-week']]
        df_copy = pd.concat([a,b],axis=1)  
        df_copy.dropna(inplace=True)

        #validation
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df_copy[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        prediction1=[]
        for i in prediction:
            prediction1.append(int(i))
   

        cat_prediction = target_encoder.inverse_transform(prediction1)

        #decoding data and save as file
        local_df = df_copy[['workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'country']]
        local_df_decode = input_feature_encoder.inverse_transform(local_df)
        local_df = pd.DataFrame(local_df_decode,columns=['workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'country'])
        local_df.reset_index(inplace=True)

        remain_col = df_copy[['age','hours-per-week']]
        remain_col.reset_index(inplace=True)
        df_copy = pd.concat([local_df,remain_col],axis=1)
        df_copy.drop('index',axis=1,inplace=True)

        

        df_copy["prediction"]=prediction1
        df_copy["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df_copy.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise SensorException(e, sys)
