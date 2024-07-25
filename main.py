import uvicorn 
import utilities
from fastapi import FastAPI , File, UploadFile , HTTPException #Query
import pickle
import numpy as np
from typing import Optional
import json
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
from utilities import SingletonLogger, Clean
import shutil
import tempfile


app = FastAPI()

class User(BaseModel):
    file: UploadFile

@app.get("/")
async def read_root():
    return {"hello": "world"}


class_model = SingletonLogger()
@app.post('/predict', status_code=201)
async def predict_flower_type(file:UploadFile):

    ''' 
        This function requests user to input a csv file of an iris dataset.
        It stores the file in tempraray storage, drops 'species' column.
        Preprocessed data is then given to pre trained model and results are printed
         
    '''
    try:
        with tempfile.NamedTemporaryFile(delete =False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_name = temp_file.name
           #temp_file_name contains path to the temp_file at this point
    finally:
        file.file.close()     
    

    df  = Clean(temp_file_name)
    ''' 
        Clean class (this class is in data_preprocess file) has a preprocess function which renames the columns of the dataset 
        and drops the Species column
    
    '''
    df = df.preprocess()

    prediction = class_model.predict(np.array(df))
    def map_values(arr):

        ''' 
            This function takes prediction (results) list as a parameter which contains three classes -1, 0, 1 
            and then it maps it on to the valid folower types and returns list of flower types predctions

        '''
        a =[]  
        for values in arr:
            if values not in arr:
                raise HTTPException(
                    status_code  =404,
                    detail ='Values not found to map',
                    headers ={"X-Error": "There goes my error"},
                )  
            else:
                if values == -1:
                  a.append ("setosa")
                elif values == 0:
                  a.append( "versicolor") 
                else:
                  a.append("virginica") 
        return a
    
    
    return map_values(prediction)

@app.get("/")
def health_check():
    return {"status": "healthy"}


if __name__ =='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)



