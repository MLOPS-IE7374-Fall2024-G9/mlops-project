# script to download data from web and api given dates, location and zone

from data import *

data_obj = Data()
start_date = "01-01-2024"
end_data = "01-02-2024"

df_map = data_obj.generate_dataset(data_obj.zones_texas, start_date, end_data)
data_obj.save_dataset(df_map, "./data.csv")
