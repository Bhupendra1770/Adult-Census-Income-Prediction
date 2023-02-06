# Adult-Census-Income-Prediction
```
step1 - data_dump.py
```
we have creat file name data_dump.py in which we have established connection with mongodb database and dump data in database.

```
step2 - utils.py
```
Here we have make some important function which will help in our component part.

```
step3 - Census 
```
In this folder we have creat our Components folder and entity folder and pipeline folder.

```
step4 - entity
```
Here we have two file one is artifact_entity.py and another is config_entity.py
config_entity.py - Here we store some important input like what all input it takes while initiating our class example in data_ingestion.py file it will take training_pipeline class which help us to make artifact directory in current directory and also store some pre define path.
artifact_entity.py - Here we store output of our component files like generally we store file path and all here.

```
step4 - Component
```

Here we have define cycle of machine learning model-
1. data_ingestion.py: In this phase we will get our data from mongodb database with the help of get_datafram_as_df function which we have define in "utils.py" and after getting data we perform some cleaning of data then we will save our data in csv file in "feature store" inside "dataingestion" directory and here we also done splitting our data in train and test so this data can further we use for data validation and transformation and all phase so we have also save our train and test data in train.py and test.py file in "dataset" directory which is inside of "dataingestion" directory inside artifact directory. 

2. data_validation.py: In this phase we will validate our new data with previous data like here we check missing columns and data drift and all , and we store this validation report in a report.yaml file and save that file in "data_validation" directory in artifact directory and path of this file save in "data_validation" class insdie artifact_entity.py file.

3. data_transformation.py: In this phase we will transform our data according to our EDA which we have done in jupyter notebook like all preprocessing steps like scalling our data encoding our target column and also convert cat column to numerical with the help of ordinal encoder and all and save all these object in our "data_transformation" directory inside artifact director and path of these object inside "DataTransformationArtifact" class in artifact_entity.py file.

4. model_trainer.py: In this phase we will train our model on random forest algorithm and save our model in artifact folder and path is saved in "ModelTrainerArtifact" class in "model_trainer" directory insdie artifact directory and path of our model stored in "model_trainer" class inside artifact_entity.py file.

5. model_evaluation.py: Here we compare our new model with previous model which is already in prodution for this we have made some function in our predictor.py file like "get_latest_model" this will fetch our currently trained model and "get_latest_save_model" this will fetch previous trained model which store in saved model directory save for transformation object and all then we will compare our both model if our currently trained model is better than previous trained model than we will go for model pusher phase and we will save our new model if not then it will gives us a error that "current trained model is not better than previous model".

6. model_pusher.py: Here we just save our new model and all transformation object which will help us in prediction.


```
step5 - pipeline
```
Here we made two file one is for training our model and onother for batch prediction-
training_pipeline.py - In this file we have just write our code in sequance like we have mention data_ingestion phase first then validation than transformation and so on.
batch_prediction.py - In this file we have just made prediction of given file and send back to prediction directory and save it their.

```
step7 - Deployement we have used docker here and with the help of github action we have made our work automate after deployement we can access our airflow so that we can shedule our training pipeline and batch_prection pipeline from their.
```
