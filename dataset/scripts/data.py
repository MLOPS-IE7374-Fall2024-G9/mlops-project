import os
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

class Data:
    def __init__(self):
        """Initializes the Data object, loading API keys and defining geographic zones."""\
        
        load_dotenv()
        
        self._DEMAND_API_KEY = os.getenv("DEMAND_API_KEY")
        self._WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

        # Check if keys are loaded
        if not self._DEMAND_API_KEY:
            raise ValueError("DEMAND_API_KEY not found in environment variables.")
        if not self._WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY not found in environment variables.")
        
        self.zones_texas = {
            "COAS": [29.749907, -95.358421],  # Houston
            "EAST": [32.351485, -95.301140],  # Tyler
            "FWES": [31.997345, -102.077915],  # Midland
            "NCEN": [32.78306, -96.80667],      # Dallas
            "NRTH": [33.913708, -98.493387],  # Wichita Falls
            "SCEN": [30.267153, -97.743057],  # Austin
            "SOUT": [26.203407, -98.230012],  # McAllen
            "WEST": [32.448736, -99.733144]   # Abilene
        }

        self.zones_new_england = {
            "4001": [43.661471, -70.255326],  # Portland, Maine
            "4002": [42.995640, -71.454789],  # Manchester, New Hampshire
            "4003": [44.475882, -73.212072],  # Burlington, Vermont
            "4004": [41.763710, -72.685097],  # Hartford, Connecticut
            "4005": [41.823989, -71.412834],  # Providence, Rhode Island
            "4006": [41.635693, -70.933777],  # New Bedford, Massachusetts (Southeast)
            "4007": [42.101483, -72.589811],  # Springfield, Massachusetts (Western/Central)
            "4008": [42.358894, -71.056742]   # Boston, Massachusetts (Northeast)
        }

    def __split_dates_monthly(self, start_date: str, end_date: str) -> list[list[str]]:
        """
        Splits the date range into monthly ranges.

        Args:
            start_date (str): The start date in "dd-mm-yyyy" format.
            end_date (str): The end date in "dd-mm-yyyy" format.

        Returns:
            list[list[str]]: A list of lists, each containing start and end dates for the month.
        """
        start = datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.strptime(end_date, "%d-%m-%Y")
        
        date_ranges = []
        current_date = start
        
        while current_date <= end:
            next_month = current_date + relativedelta(months=1)
            last_day_of_month = next_month - relativedelta(days=1)
            
            end_of_range = min(last_day_of_month, end)
            date_ranges.append([current_date.strftime("%Y-%m-%d"), end_of_range.strftime("%Y-%m-%d")])
            
            current_date = next_month
        
        return date_ranges
    
    def __split_dates_yearwise(self, start_date: str, end_date: str) -> list[list[str]]:
        """
        Splits the date range into yearly ranges.

        Args:
            start_date (str): The start date in "dd-mm-yyyy" format.
            end_date (str): The end date in "dd-mm-yyyy" format.

        Returns:
            list[list[str]]: A list of lists, each containing start and end dates for the year.
        """
        start_year = datetime.strptime(start_date, "%d-%m-%Y").year
        end_year = datetime.strptime(end_date, "%d-%m-%Y").year
        
        date_ranges = []
        
        for year in range(start_year, end_year + 1):
            if year == start_year:
                year_start = start_date  # Use the specified start date for the first year
            else:
                year_start = f"01-01-{year}"  # First day of the year for subsequent years
                
            year_end = f"31-12-{year}" if year < end_year else end_date  # End of the year or specified end_date
            date_ranges.append([year_start, year_end])
        
        return date_ranges

    def get_demand_data(self, subba: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches demand data from the EIA API for the specified sub-balance area (subba).

        Args:
            subba (str): The sub-balance area identifier.
            start_date (str): The start date in "dd-mm-yyyy" format.
            end_date (str): The end date in "dd-mm-yyyy" format.

        Returns:
            pd.DataFrame: A DataFrame containing the demand data, or an empty DataFrame if the request fails.
        """
        print("Getting demand data")
        print(start_date, end_date)
        
        start_date = datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")

        demand_url = (
            "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/?api_key={0}"
            "&data[]=value&facets[subba][]={1}&start={2}&end={3}"
        ).format(self._DEMAND_API_KEY, subba, start_date, end_date)

        response = requests.get(demand_url)

        if response.status_code == 200:
            json_data = response.json()
            df_demand = pd.DataFrame(json_data["response"]["data"])
            df_demand = df_demand.drop(columns=['subba', 'parent', 'parent-name'])
            return df_demand

        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()  # Return an empty DataFrame

    def get_weather_data(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical weather data for the specified location.

        Args:
            location (str): The location coordinates or name.
            start_date (str): The start date in "dd-mm-yyyy" format.
            end_date (str): The end date in "dd-mm-yyyy" format.

        Returns:
            pd.DataFrame: A DataFrame containing the weather data, or an empty DataFrame if the request fails.
        """
        print("Getting weather data")
        monthly_date_ranges = self.__split_dates_monthly(start_date, end_date)
        
        all_dataframes = []

        for start, end in monthly_date_ranges:    
            print("Getting data for dates", start, end)

            weather_url = (
                "https://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={0}&q={1}"
                "&format=json&date={2}&enddate={3}&tp=1"
            ).format(self._WEATHER_API_KEY, location, start, end)
            response = requests.get(weather_url)
            if response.status_code == 200:
                json_data = response.json()
                df_weather = pd.DataFrame(json_data["data"]["weather"])
                all_dataframes.append(df_weather)
            else:
                print(f"Failed to retrieve data for {start} to {end}: {response.status_code}")
        
        # Combine all DataFrames into a single DataFrame
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()  # Return an empty DataFrame

    def process_weather_data(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the weather data DataFrame into a standardized format.

        Args:
            df_weather (pd.DataFrame): The DataFrame containing weather data.

        Returns:
            pd.DataFrame: A processed DataFrame with relevant weather information.
        """
        processed_data = []
        for index, row in df_weather.iterrows():
            date = row["date"]
            hourly_data = row["hourly"]

            for hour in hourly_data:
                time = hour["time"]
                time = time.zfill(4)
                hour_of_day = time[:2]

                datetime_str = f"{date}T{hour_of_day}"
                
                hour["datetime"] = datetime_str
                processed_data.append(hour) 

        df_processed = pd.DataFrame(processed_data)
        cols = ['datetime'] + [col for col in df_processed.columns if col != 'datetime']
        df_processed = df_processed[cols]
        df_processed = df_processed.drop(columns=['time', 'tempC', 'windspeedKmph', 'weatherIconUrl', 'weatherDesc', 'winddirDegree', 'winddir16Point'])

        return df_processed
    
    def generate_dataset(self, zones: dict, start_date: str, end_date: str) -> dict:
        """
        Generates a dataset by fetching and merging demand and weather data for specified zones.

        Args:
            zones (dict): A dictionary of zones with their coordinates.
            start_date (str): The start date in "dd-mm-yyyy" format.
            end_date (str): The end date in "dd-mm-yyyy" format.

        Returns:
            dict: A dictionary mapping zone names to their respective merged DataFrames.
        """
        api_calls = 0
        dates = self.__split_dates_yearwise(start_date, end_date)
        df_map = {}

        for zone in zones:
            if api_calls >= 500:
                break
            
            print(zone)
            zone_name = zone
            date_df_list = []

            for date in dates:
                start = date[0]
                end = date[1] 
                print(start, end)

                df_demand = self.get_demand_data(zone, start, end)

                if df_demand.empty:
                    print("Demand data not fetched for" + zone + " " + start + " " + end)
                    continue
                
                city_location = ','.join(map(str, zones[zone]))
                df_weather = self.get_weather_data(city_location, start, end)
                
                if df_weather.empty:
                    print("Weather data not fetched for" + zone + " " + start + " " + end)
                    continue
                
                df_weather = self.process_weather_data(df_weather)
                
                df_weather.rename(columns={'datetime': 'datetime'}, inplace=True)
                df_demand.rename(columns={'period': 'datetime'}, inplace=True)

                df_merged_dataset = pd.merge(df_weather, df_demand, on='datetime', how='inner')
                df_merged_dataset['zone'] = zone_name

                date_df_list.append(df_merged_dataset)
                
                api_calls += 12
            
            combined_df = pd.concat(date_df_list, ignore_index=True)
            df_map[zone] = combined_df
        return df_map
    
    def save_dataset(self, df_map: dict, path: str) -> None:
        """
        Saves the combined dataset to a CSV file.

        Args:
            df_map (dict): A dictionary mapping zone names to their respective DataFrames.
            path (str): The path where the CSV file will be saved.

        Returns:
            None
        """
        df_combined = pd.concat(df_map.values(), ignore_index=True)
        df_combined.sort_values(by='datetime', inplace=True)
        df_combined.to_csv(path, index=False)
