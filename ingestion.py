import duckdb
import pandas as pd

def load_data():

    conn = duckdb.connect('./duckdb/file.duckdb')
    with open("./sql/ingestion.sql", "r") as f:
        sql_content = f.read()
    
    conn.sql(sql_content)

def create_analytics_master():

    conn = duckdb.connect('./duckdb/file.duckdb')
    train = conn.sql('SELECT * FROM train').df()
    print("train.shape", train.shape)
    test = conn.sql('SELECT * FROM test').df()
    print("test.shape", test.shape)

    test["price"] = -1

    cols = train.columns.tolist()
    test = test[cols]
    concat_df = pd.concat([train, test], ignore_index=True)
    print("concat_df.shape", concat_df.shape)

    properties = conn.sql('SELECT * FROM properties').df()
    geo_data = conn.sql('SELECT * FROM geo_attributes').df()

    analytics_master = pd.merge(concat_df, properties, on='property_key', how='left')
    analytics_master = pd.merge(analytics_master, geo_data, on=['district', 'street', 'project'], how='left')
    print(analytics_master[:5])
    print(analytics_master["num_mrt_stations_500m"].isnull().sum())

    conn.sql("CREATE TABLE analytics_master AS SELECT * FROM analytics_master")
    print("done.")

def main():

    load_data()
    create_analytics_master()

if __name__ == "__main__":
    
    main()