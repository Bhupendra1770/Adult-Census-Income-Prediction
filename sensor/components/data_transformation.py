from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector



class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            

###################################################################################################################################

            #using labelencoder converting cat columns to numerical
            #cat_col=[]
            #for i in input_feature_train_df.columns:
                #if input_feature_train_df[i].dtype=='object':
                    #cat_col.append(i)

            #label_encoder1 = preprocessing.LabelEncoder()
            #for i in cat_col:
                #input_feature_train_df[i]= label_encoder1.fit_transform(input_feature_train_df[i])
            #print(input_feature_train_df.head())
            #for i in cat_col:               
                #input_feature_test_df[i]= label_encoder1.fit_transform(input_feature_test_df[i])
 
            

            #selecting target feature for train and test dataframe
            #target_feature_train_df = train_df[TARGET_COLUMN]
            #target_feature_test_df = test_df[TARGET_COLUMN]

            #print(input_feature_train_df)
            #print('###################################')
            ##print(input_feature_test_df)
            #print('########################################################')
            #print(target_feature_train_df)
#####################################################################################################################################
            train_df1 = train_df.copy()
            train_df1.drop('salary',axis=1,inplace=True)
            encoder = OrdinalEncoder()   #object
            #data.drop('salary',axis=1,inplace=True)
            cat_col=[]
            for i in train_df1.columns:
                if train_df1[i].dtype=='object':
                    cat_col.append(i)
            data_categorical = train_df1[cat_col]


            data_encoded = encoder.fit_transform(data_categorical)
            a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = train_df[['age','hours-per-week','salary']]
            #c=data['hours-per-week']
            train_df = pd.concat([a,b],axis=1)   #train df

  
            #for test_df
            test_df1 = test_df.copy()
            test_df1.drop('salary',axis=1,inplace=True)
            cat_col=[]
            for i in test_df1.columns:
                if test_df1[i].dtype=='object':
                    cat_col.append(i)
            data_categorical = test_df1[cat_col]


            data_encoded = encoder.transform(data_categorical)

            a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = test_df[['age','hours-per-week','salary']]
            #c=data['hours-per-week']
            test_df = pd.concat([a,b],axis=1)     #test df
  

  
  ##########################################################################################################################
  
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder2 = LabelEncoder()
            #label_encoder2.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder2.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder2.transform(target_feature_test_df)

            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder2)

            utils.save_object(file_path=self.data_transformation_config.input_feature_encoder_path, obj=encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path,
                input_feature_encoder_path = self.data_transformation_config.input_feature_encoder_path
                
                )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)


