CREATE TABLE cpi AS SELECT * FROM read_csv_auto('./data/cpi.csv');

CREATE TABLE geo_attributes AS SELECT * FROM read_csv_auto('./data/geo_attributes.csv');

CREATE TABLE interest AS SELECT * FROM read_csv_auto('./data/interest.csv');

CREATE TABLE properties AS SELECT * FROM read_csv_auto('./data/properties.csv');

CREATE TABLE rentIndex AS SELECT * FROM read_csv_auto('./data/rentIndex.csv');

CREATE TABLE test AS SELECT * FROM read_csv_auto('./data/test.csv');

CREATE TABLE train AS SELECT * FROM read_csv_auto('./data/train.csv');

CREATE TABLE vacant AS SELECT * FROM read_csv_auto('./data/vacant.csv');