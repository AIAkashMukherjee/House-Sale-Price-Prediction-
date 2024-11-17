from src.entity.config_entity import DataTransformationConfig
from src.exceptions.expection import CustomException
from src.logger.custom_logging import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import sys
import numpy as np
from src.utils.utlis import *


class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        logger.info('Data Transformation started')
        self.config=config

    def get_data_transformation(self):
        try:
            logger.info('In funciton')

            numerical_features=['Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond',
        'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1',
        'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
        '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
        'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
        'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt',
        'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF',
        'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val',
        'Mo Sold', 'Yr Sold']
            
            string_column=['MS Zoning', 'Street', 'Lot Shape', 'Land Contour', 'Utilities',
        'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
        'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
        'Exterior 1st', 'Exterior 2nd', 'Exter Qual', 'Exter Cond',
        'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
        'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating', 'Heating QC',
        'Central Air', 'Electrical', 'Kitchen Qual', 'Functional',
        'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond',
        'Paved Drive', 'Sale Type', 'Sale Condition']
            
            cat_pipeline=Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),  
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            num_pipeline=Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy = 'mean')),
                ("scaler", StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('Num_pipeline', num_pipeline,numerical_features),
                ('Cat_pipeline', cat_pipeline,string_column),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
        

    def initate_data_transformation(self):
        train_path=self.config.train_file_path
        test_path=self.config.test_file_path
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            # print(train_data)
            # print(test_data.columns)

            # train_data.dropna(inplace=True)
            # test_data.dropna(inplace=True)

            preprocessor_obj=self.get_data_transformation()

            target_columns='SalePrice'
            drop_columns=[ target_columns,'Order','PID','MS SubClass','Fence','Fireplace Qu','Mas Vnr Type','Alley','Pool QC','Misc Feature']

            logger.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            # print(input_feature_train_data)
            traget_feature_train_data = train_data[target_columns]
            

            logger.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            traget_feature_test_data = test_data[target_columns]

            # Apply preprocessor object on our train data and test data
            input_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

            print(input_train_arr)
            # print(traget_feature_train_data)

             # Apply preprocessor object on our train data and test data
            train_array = np.c_[input_train_arr, traget_feature_train_data.values.reshape(-1, 1)]



            test_array=np.c_[input_test_arr,np.array(traget_feature_test_data)]

            save_obj(file_path=self.config.preprocessor_obj,obj=preprocessor_obj)

            train_df = pd.DataFrame(train_array)
            test_df = pd.DataFrame(test_array)

            train_df.to_csv(self.config.save_train_path, index=False)
            test_df.to_csv(self.config.save_test_path, index=False)

            # return (train_array,
            #         test_array,
            #         self.config.preprocessor_obj)



        except Exception as e:
            raise CustomException(e,sys)    