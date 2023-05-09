import duckdb
import pandas as pd
import numpy as np
import datetime
import os

import re
from toolz import *
from toolz.curried import *

from pycaret.regression import RegressionExperiment
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

conn = duckdb.connect('./duckdb/file.duckdb')

def clean_tenure(rows):

    if rows["tenure"] == "Freehold":
        return 99
    else:
        return rows["contract_year"] - int(rows["tenure"][-4:])

def yq2ym(df):
    """
    convert year-quater in string format to monthly period
    """

    # repeat the first row for desired interpolation result
    df = pd.concat([df.head(0), df])
    df.at[0, "date"] = "2023 2Q "

    df["date"] = (pd.to_datetime((df["date"].apply(
                                     lambda s:
                                        re.sub(r"(\d+) (\d)Q ", r"\1-Q\2", s)
                                   )))
                  .dt
                  .to_period('M'))
    df = df.set_index("date").resample("M", convention = "end").interpolate("linear")        
    return df

def ym2ym(df):
    """
     convert year-month in string format to monthly period
    """
    df["date"] = (pd.to_datetime(df["date"], format = "%Y %b ")
                  .dt
                  .to_period('M'))
    df = df.set_index("date")
    return df

def mergeDfs(dfs, on):
    """
    join dataframes into one
    """
    df = reduce(partial(pd.merge, on = on, how = "inner"), dfs)
    return df

def get_processed_cpi():
    
    cpi_df = conn.sql('SELECT * FROM cpi').df()
    cpi_df = cpi_df.rename({"Data Series": "date"}, axis=1)
    recent_cpi = cpi_df[:2]["CPI"].tolist()
    rate_of_change = recent_cpi[0] / recent_cpi[1]

    cpi_2023_Jan = recent_cpi[0] * rate_of_change
    cpi_2023_Feb = np.ceil(cpi_2023_Jan * rate_of_change)
    cpi_2023_Mar = np.ceil(cpi_2023_Feb * rate_of_change)

    _new = pd.DataFrame({
                'date': ['2023 Jan ', '2023 Feb ', '2023 Mar '],
                'CPI' : [cpi_2023_Jan, cpi_2023_Feb, cpi_2023_Mar]   # 'CPI' : [111.397, 112, 113]
    })

    cpi_df = pd.concat([cpi_df, _new])
    cpi_df_m = ym2ym(cpi_df)

    return cpi_df_m

def get_processed_interest_rates():

    ### Based on https://www.smart-towkay.com/blog/view/307-will-singapore-home-loan-interest-rate-reach-6-in-2023

    interest_df = conn.sql('SELECT * FROM interest').df()
    interest_df = interest_df.rename({"Data Series": "date"}, axis=1)
    recent_interest = interest_df[:2]["InterestRate"].tolist()
    rate_of_change = recent_interest[0] / recent_interest[1]

    ### Take average betwen the forecast interest rate (from above url)
    ### and December interest rate
    interest_2023_Jan = (recent_interest[0] + 6) / 2 
    interest_2023_Feb = interest_2023_Jan * rate_of_change  ## adjusted with rate of change
    interest_2023_Mar = interest_2023_Feb * rate_of_change  ## adjusted with rate of change

    _new = pd.DataFrame({
                'date': ['2023 Jan ', '2023 Feb ', '2023 Mar '],
                'InterestRate' : [interest_2023_Jan, interest_2023_Feb, interest_2023_Mar]
    })

    interest_df = pd.concat([interest_df, _new])
    interest_df_m = ym2ym(interest_df)

    return interest_df_m

def get_processed_rental_index():

    rentIndex_df = conn.sql('SELECT * FROM rentIndex').df()
    rentIndex_df = rentIndex_df.rename({"Data Series": "date"}, axis=1)
    recent_rentIndex = rentIndex_df[:2]["RentIndex"].tolist()
    rate_of_change = recent_rentIndex[0] / recent_rentIndex[1]

    _new = pd.DataFrame({
                'date': ['2023 1Q '],
                'RentIndex' : [recent_rentIndex[0] * rate_of_change]
    })

    rentIndex_df = pd.concat([_new, rentIndex_df]).reset_index(drop=True)
    rentIndex_df_m = yq2ym(rentIndex_df)

    return rentIndex_df_m

def get_processed_vacancy():

    vacant_df = conn.sql('SELECT * FROM vacant').df()
    vacant_df = vacant_df.rename({"Data Series": "date"}, axis=1)
    recent_vacancy = vacant_df.loc[0][["Available", "Vacant"]].tolist()

    _new = pd.DataFrame({
                'date': ['2023 1Q '],
                'Available': [recent_vacancy[0]],
                'Vacant': [recent_vacancy[1]]   ## 530
    })

    vacant_df = pd.concat([_new, vacant_df]).reset_index(drop=True)
    vacant_df_m = yq2ym(vacant_df)

    return vacant_df_m

def get_master_analytics_data():

    master_df = conn.sql('SELECT * FROM analytics_master').df()
    print(master_df.shape)

    master_df["source"] = master_df["price"].apply(lambda p: "train" if p >= 0 else "test")
    master_df["to_remove"] = master_df["lat"].isnull()
    master_df = master_df[master_df["to_remove"] == False]
    print(master_df.shape)

    return master_df

def get_ura_data(rootdir="./ura_data"):

    appended_data = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if ('2018' in filepath) | ('2019' in filepath) | ('2020' in filepath)| ('2021' in filepath)| ('2022' in filepath) | ('2023' in filepath):
                # store DataFrame in list
                print(filepath)
                data = pd.read_csv(filepath, encoding='ISO-8859-1')
                appended_data.append(data)

    ura_df = pd.concat(appended_data)
    ura_df = ura_df[['Project Name', 'Sale Date', 'Planning Region', 'Planning Area']]

    return ura_df

def join_all_tables():

    cpi_df_m = get_processed_cpi()
    interest_df_m = get_processed_interest_rates()
    rentIndex_df_m = get_processed_rental_index()
    vacant_df_m = get_processed_vacancy()

    df = mergeDfs([cpi_df_m, interest_df_m, rentIndex_df_m, vacant_df_m], on = "date") 
    df.index = [ str(i) for i in df.index ]

    master_df = get_master_analytics_data()
    ura_df = get_ura_data(rootdir="./ura_data")

    master_df["date"] = master_df["contractDate"].apply(
                            lambda d: d.strftime("%Y-%m")
                        )
    master_df = master_df.set_index("date")
    master_df = pd.merge(master_df, df, left_index=True, right_index=True, how="left")

    master_df["contract_month_str"] = [ d[-2:] for d in master_df.index ]
    master_df["contract_month"] = [ int(d[-2:]) for d in master_df.index ]
    master_df["contract_year"] = [ int(d[:4]) for d in master_df.index ]

    ura_df['Sale Date'] = ura_df['Sale Date'].apply(lambda x: x.replace('-22', ' 2022'))
    ura_df['Sale Date'] = ura_df['Sale Date'].apply(lambda x: x.replace("-", " "))
    ura_df['Sale Date'] = ura_df['Sale Date'].apply( lambda x : 
                                    datetime.datetime.strptime(x, "%d %b %Y").strftime("%Y-%m-%d"))
    ura_df['Sale Date'] = pd.to_datetime(ura_df['Sale Date'])

    master_df["tenure_cleaned"] = master_df.apply(lambda t: clean_tenure(t), axis=1)

    master_ura = pd.merge(master_df, ura_df, how='left', 
                        left_on=['project', 'contractDate'], 
                        right_on=['Project Name', 'Sale Date']) 
    
    return master_ura

def main():
    
    identifiers = ["property_key", "source", "contractDate"]
    target = ["price"]
    numerical = ["lat", "lng", "area", "tenure_cleaned", "num_schools_1km", "num_supermarkets_500m", 
                "num_mrt_stations_500m", "CPI", "InterestRate", "RentIndex", "Available", "Vacant"]
    categorical = ["contract_month_str", "district", "floorRange", "propertyType", "typeOfArea", "marketSegment",
                  "Planning Region", "Planning Area"]
    
    master_ura = join_all_tables()
    analytics_df = master_ura[ identifiers + target + numerical + categorical ]
    analytics_df = pd.get_dummies(analytics_df, columns=categorical)
    
    train_df = analytics_df[analytics_df["source"]=="train"].reset_index(drop=True)
    test_df = analytics_df[analytics_df["source"]=="test"].reset_index(drop=True)
    train_df.shape, test_df.shape

    groundtruth_df = train_df[identifiers]
    train_df = train_df[ [c for c in train_df.columns if c not in identifiers] ]
    cols = [c for c in train_df.columns if c != "price"]

    train_df = train_df[cols + ["price"]]

    groundtruth_test = test_df[identifiers]
    test_df = test_df[ [c for c in test_df.columns if c not in identifiers + ["price"]] ]
    test_df = test_df[cols]

    x_train, x_val, _, _ = train_test_split(train_df, groundtruth_df, 
                                            test_size=0.25, random_state=1108
                           )

    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)

    ### Modelling
    regExp = RegressionExperiment()
    s = regExp.setup(data=x_train, target='price', test_data=x_val,
                     session_id=42, log_experiment=False, preprocess=False,
                     numeric_features=cols, remove_multicollinearity=False, 
                     fold=4)
    
    prep_pipeline = regExp.get_config("pipeline")
    x_train_prep = prep_pipeline.transform(x_train)
    x_val_prep = prep_pipeline.transform(x_val)

    x_val_prep = x_val_prep.drop(columns="price")
    eval_set = [(x_val_prep, x_val["price"])]

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6, 
                       seed=1108)
    fit_kwargs={
        "early_stopping_rounds": 10,
        "eval_metric": "rmse",
        "eval_set": eval_set
    }

    xgb_p = regExp.create_model(xgb, fit_kwargs=fit_kwargs)
    xgb_tuned = regExp.tune_model(xgb_p)

    test_df = regExp.predict_model(xgb_tuned, data=test_df)
    predictions = pd.DataFrame({
                    "property_key": groundtruth_test["property_key"].tolist(),
                    "contractDate": groundtruth_test["contractDate"].tolist(),
                    "prediction": test_df["prediction_label"].tolist()
    })

    test_raw = conn.sql('SELECT * FROM test').df()
    predictions = pd.merge(test_raw, predictions, 
                           on=["property_key", "contractDate"], how="left")
    
    predictions["prediction"] = predictions["prediction"].fillna( predictions["prediction"].mean() )
    predictions.to_csv("./outputs/submission.csv", index=False)

if __name__ == "__main__":
    
    main()