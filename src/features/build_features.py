
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

#Handling missing values in training data
def house_price_cleaner(data):
    data = data.copy()
    
    drop_cols = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    
    fill_none_cols = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                      'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
                      'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    
    fill_zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                      'TotalBsmtSF', 'GarageCars', 'GarageArea'] 
    
    fill_median_cols = ['GarageYrBlt']
    
    fill_mode_cols = ['MSSubClass', 'MSZoning', 'Electrical', 'KitchenQual', 'Functional', 'SaleType'] 
    

    #Drop columns which are having high missing values
    data = data.drop(drop_cols, axis = 1)    
    
    #Fill None for null values
    for c4 in fill_none_cols:
        if c4 in data.columns:
            data[c4] = data[c4].fillna("None")
    
    #Filling Zero at null values
    for c5 in fill_zero_cols:
        if c5 in data.columns:
            data[c5] = data[c5].fillna(0)
    
    
    #Fill median null values
    for c1 in fill_median_cols:
        if c1 in data.columns:
            data[c1] = data[c1].fillna(data[c1].median())
        
    #Apply mode for null values
    for c6 in fill_mode_cols:
        if c6 in data.columns:
            data[c6] = data[c6].fillna(data[c6].mode().iloc[0])
 
    return data

#Cleaning null values in LotFrontage column by filling median value of each neighborhood and then filling remaining null values with global median value
def frontage_cleaning(data):
    data = data.copy()
    
    neigh_median = data.groupby("Neighborhood")["LotFrontage"].median()
    
    global_median = data["LotFrontage"].median()
    
    data["LotFrontage"] = data["LotFrontage"].fillna(
        data["Neighborhood"].map(neigh_median)
    )
    data["LotFrontage"] = data["LotFrontage"].fillna(global_median)
    
    return data

#Outlier removal using IQR method
def clip_outliers(df):
    df = df.copy()

    cols_for_outlier_check = ["GrLivArea", "TotalBsmtSF", "GarageArea", "1stFlrSF", "LotArea"]
    for col in cols_for_outlier_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = df[col].clip(lower, upper)
    return df

clipper = FunctionTransformer(
    clip_outliers, 
    validate = False
)

#Feature engineering
def house_price_feature_engineering(data):
    data = data.copy()
    
    data["TotalSF"] = (data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"])
    
    data["TotalBathrooms"] = (data["FullBath"] + 0.5 * data["HalfBath"] + data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"])
    
    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
    data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]
    
    data["Qual_GrLivArea"] = data["OverallQual"] * data["GrLivArea"]
    
    log_features = ["LotArea","GrLivArea","TotalSF","MasVnrArea"]
    
    
    for feat in log_features:
        data[f"Log_{feat}"] = np.log1p(data[feat])

    cols_to_drop = ["TotalBsmtSF","1stFlrSF","2ndFlrSF","FullBath","HalfBath","BsmtFullBath",
                    "BsmtHalfBath","YearBuilt","YearRemodAdd","GarageYrBlt","YrSold","MoSold"
                   ]

    data = data.drop(columns=cols_to_drop, errors="ignore")
    return data

feature_engineer = FunctionTransformer(
    house_price_feature_engineering, 
    validate = False
    )


#Preprocessing pipeline
def build_preprocessor(data):
    data = data.copy()
    numerical_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(exclude = ['int64', 'float64']).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer' , SimpleImputer(strategy = "median")),
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown= "ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )
    return preprocessor
