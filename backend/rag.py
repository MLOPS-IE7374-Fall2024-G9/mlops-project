from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import PromptTemplate
import chromadb

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
import google.generativeai as genai

import yfinance as yf
import requests
import logging 
import warnings
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
warnings.filterwarnings("ignore")

from model.scripts.inference import ModelInference
from dataset.scripts.data import DataCollector
from backend.utils import *

# ------------------------------------
logger = logging.getLogger("RAG Backend")
# Check if the logger already has handlers
if not logger.hasHandlers():
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # Create a stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(ch)
# ------------------------------------

class RAG:
    def __init__(self, llm_name="llama3-groq-tool-use", embed_name="BAAI/bge-small-en"):
        load_dotenv()
        
        self.llm_name = llm_name
        self.embed_name = embed_name
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm_model = genai.GenerativeModel('gemini-pro')
        
        # prompts
        self.system_prompt = """You are an AI financial analyst specializing in analyzing energy demand forecasting and its impact on the stock performance of energy companies based on a given location.

                                Key Capabilities:
                                1. Energy Demand Analysis: Provide insights into energy demand forecasting based on weather patterns and location-specific factors.
                                2. Stock Market Impact: Predict how weather and energy demand affect stock prices of energy-related companies in the specified city or region.

                                Guidelines:
                                - Responding to Queries:
                                1. If a query is related to energy forecasting, weather, or stock market analysis, follow the detailed step-by-step process described below to provide a comprehensive report.
                                2. For unrelated queries, use `query_tool` to answer basic questions. Redirect the user to focus on energy demand and stock predictions by suggesting questions in your domain of expertise.

                                - City-Specific Report:
                                For any query involving a city or region:
                                - Always include weather information, energy demand, and possible stock impact in the report.
                                - Use the `Get Stock Historical Price` tool automatically for companies associated with that location to assess historical trends.
                                - Provide an analysis of stock trends categorized as increase, decrease, or constant.

                                Steps for Energy Demand and Stock Prediction:
                                1. Identify the Location:
                                - Extract the city or region from the query.
                                - Use the location to fetch weather, energy demand, and stock data.

                                2. Fetch Weather Data:
                                - Use the "Get Weather Data" tool to gather weather details (e.g., temperature, precipitation, wind).
                                - Identify how weather impacts energy demand.

                                3. Fetch Energy Demand Data:
                                - Use the "Get Energy Demand Data" tool to predict energy demand based on weather, seasonal factors, and location.

                                4. Fetch Stock Historical Data:
                                - Use the "Get Stock Historical Details" tool to analyze the historical performance of the stocks effected around the location.
                                - Determine how stocks have reacted to similar energy demand and weather patterns.

                                6. Fetch Financial Statements:
                                - Use the "Get Financial Statements" tool for deeper insight into the financial health of the companies around the location.

                                7. Analyze the Impact:
                                - Based on weather, energy demand, and financial data, evaluate how the forecast will influence the stocks of identified companies.

                                8. Predict Stock Movement for each company:
                                - Provide a detailed analysis with categories:
                                    - Increase: Stocks are likely to rise.
                                    - Decrease: Stocks are likely to fall.
                                    - Constant: Minimal change expected.

                                Response Format:
                                For energy demand and stock analysis:
                                ```plaintext
                                Question: <Input question>
                                Thought: <Initial thoughts and plan>
                                Action: <Tool or action to perform>
                                Action Input: <Input for the action>
                                Observation: <Result of the action>
                                ... (Repeat Action/Observation steps as needed)
                                Thought: I now have the final analysis.
                                Final Answer:
                                Weather Report: <Summary of weather data>
                                Energy Demand: <Energy demand forecast and reasoning>
                                Stock Analysis:
                                - Company 1: <Expected trend and reasoning>
                                - Company 2: <Expected trend and reasoning>
                                Recommendation: <Buy, Sell, or Hold> with reasons.
                                """
        
        self.react_system_prompt = PromptTemplate("""
                                    You are an AI financial analyst with the ability to analyze how external factors, like weather, can affect energy demand and the performance of companies in the energy sector. Based on the given query, you will perform a series of steps to gather relevant data, analyze it, and predict potential stock fluctuations for companies that may be impacted by the demand forecasting in the region.

                                    Every time, first identify the location and then use the following tools:

                                    1) **Get Weather Data**: Use this tool to fetch the weather forecast for the given location, such as temperature, precipitation, wind speed, etc. This data will help understand the likely impact of weather conditions on energy demand.
                                    2) **Get Energy Demand Data**: Use this tool to fetch energy demand forecast data for the location, considering factors such as time of year, weather conditions, and general energy usage trends.
                                    3) **Get Stock Historical Price**: Use this tool to gather historical stock data for companies in the energy sector that are located in or near the given region. This will help in assessing how the companies have historically reacted to changes in energy demand or weather events.
                                    4) **Get Financial Statements**: Use this tool to retrieve financial statements for the energy companies that are found to be impacted by changes in energy demand due to weather conditions.
                                    5) **Predict Stock Fluctuations**: Based on the gathered data, predict how the weather forecast and energy demand will influence the stock performance of companies in the region. Provide a detailed analysis including numbers and reasons to justify the prediction.

                                    Steps:

                                    Note- If any step fails, simply move to the next one.

                                    1) **Get the Location Name**: Identify the location from the query and use it to fetch the weather and energy demand data.
                                    2) **Fetch Weather Data**: Use the "Get Weather Data" tool to understand the weather forecast for the location and identify how it might affect energy demand.
                                    3) **Fetch Energy Demand Data**: Use the "Get Energy Demand Data" tool to understand the expected energy demand based on weather patterns, location, and other factors.
                                    4) **Identify Companies in the Region**: Using the location, identify the companies in the energy sector that may be affected by changes in energy demand due to weather.
                                    5) **Fetch Stock Historical Data**: Use the "Get Stock Historical Price" tool to get stock data for the identified companies to assess how they have reacted to similar past conditions.
                                    6) **Get Financial Statements**: Use the "Get Financial Statements" tool to fetch financial details for the companies and understand how they have performed in response to similar weather or demand events.
                                    7) **Analyze the Impact**: Using all the gathered data, analyze how the weather forecast and energy demand are likely to impact the stock performance of the relevant companies.
                                    8) **Predict Stock Movement**: Based on your analysis, predict the potential stock fluctuations, whether to buy, hold, or sell.

                                    Use the following format:

                                    Question: the input question you must answer
                                    Thought: your initial thought, detailing how you will process the question and the tools to use
                                    Action: the action to take, should be one of [Get Weather Data, Get Energy Demand Data, Get Stock Historical Price, Get Financial Statements]
                                    Action Input: the input for the action
                                    Observation: the result of the action
                                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                                    Thought: I now know the final answer
                                    Final Answer: the final recommendation, which could include the expected stock fluctuation prediction (buy, hold, sell) with reasoning based on the gathered data

                                    Begin!

                                    Question: {input}
                                    Thought:{agent_scratchpad}

                                    """)
        
        self.name_extraction_prompt = """You are a location extraction bot. Your task is to extract location names strictly from input queries following these rules and steps:
                                    Do not cross question just extract the name. Also if you think the name of the city location spelling is wrong, update it to whatever you think is the closest and extract it.
                                    ### Steps:

                                    1. **Determine if the query is a greeting**:  
                                    - If the input is a greeting (e.g., "Hi," "Hello," "Good morning"), immediately return `"none"`.  
                                    - Do not analyze or extract anything further if it's a greeting.

                                    2. **Check if the query is related to weather or energy demand forecasts**:  
                                    - If the query is **not** related to weather or energy demand, immediately return `"none"`.  
                                    - Do not attempt to extract anything further.

                                    3. **Look for a location name in the query**:  
                                    - If a location name is mentioned, extract it directly from the query and return it as a **single string**.  
                                    - Only return the name of the location (e.g., city, state, or region). Avoid sentences, explanations, or additional details.

                                    4. **If no location name is found in the query**:  
                                    - Return `"none"` as a single word.  
                                    - Do not include any additional text or context in the response.

                                    ### Rules:

                                    - Always extract strictly from the input query only.  
                                    - Do **not** use prior information, assumptions, or logic to infer a location.  
                                    - If the query does not explicitly mention a location or is unrelated to weather/energy demand, always return `"none"`.  
                                    - Ensure the output is only the name of the location or `"none"`.  

                                    ### Examples:

                                    1. **Input**: "What is the weather forecast for Boston?"  
                                    **Output**: `"Boston"`

                                    2. **Input**: "Hi, how are you?"  
                                    **Output**: `"none"`

                                    3. **Input**: "Tell me the energy demand for California."  
                                    **Output**: `"California"`

                                    4. **Input**: "What is the temperature?"  
                                    **Output**: `"none"`

                                    5. **Input**: "Hello!"  
                                    **Output**: `"none"`
                                    """
                                    

        self.db_path = "./chromadb"
        self.collection_name = "documents"
        self.messages = [ChatMessage(
                role="system", content=self.system_prompt
            )]
        self.tools = []

        self.GEO_API_KEY = os.getenv("GEO_API_KEY")
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"

        # data - for weather data
        self.data_obj = DataCollector()
        
        # model - for demand prediction
        self.model_inference = ModelInference()
        self.model_inference.download_model()
        self.model_inference.load_model()

        # inits
        self.init_models()
        # self.init_vector_db()
        self.init_tools()
        self.init_agent()

    # --------------------------------------------
    def init_models(self):
        self.llm = Ollama(model=self.llm_name, request_timeout=300.0, system=self.system_prompt)

        # get embed
        self.embed_model = HuggingFaceEmbedding(model_name=self.embed_name)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        logger.info("Init models")

    def init_vector_db(self):
        self.client = chromadb.PersistentClient(path=self.db_path)
        logger.info("Init vector db")

    def read_documents(self, path):
        reader = SimpleDirectoryReader(input_dir=path)
        documents = reader.load_data()
        logger.info("Read documents successfully")

        return documents
    
    def get_storage_context(self):
        collection = self.client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store, storage_context
    
    def init_query_engine(self):
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=self.__similarity)
        self.chat_engine = self.index.as_chat_engine(chat_mode="context", verbose=True)
        logger.info("Init query engines")
    
    # ---------------------------------
    # tools 
    def get_coordinates(self, location_name: str) -> str:
        """
        Gets the GEO coordinates for the given location.
        
        Args:
            location_name (str): The name of the location to get coordinates for.
            
        Returns:
            str: A string in the format "latitude,longitude" representing the coordinates of the location.
        """
        # Init params
        params = {
            "q": location_name,
            "key": self.GEO_API_KEY,
        }

        try:
            # API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()

            # Check and return
            if data and "results" in data and len(data["results"]) > 0:
                coordinates = data["results"][0]["geometry"]
                return f"{coordinates['lat']},{coordinates['lng']}"
            else:
                print("Location not found.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def predict_energy_demand(self, location_name: str) -> float:
        """
        Predicts and returns the energy demand in MwH for the given location.
        
        Args:
            location_name (str): The name of the location to predict energy demand for.
            
        Returns:
            float: The predicted energy demand in MwH.
        """
        # Get coordinates
        coordinates = self.get_coordinates(location_name) 

        # Get the weather
        input_df = self.model_inference.get_weather_data(location=coordinates)

        # Predict energy demand
        predictions = self.model_inference.predict(input_df)
        return predictions

    def get_weather_information_today(self, location_name: str) -> str:
        """
        Returns today's weather information for a given location in a formatted string.
        
        Args:
            location_name (str): The name of the location to retrieve weather data for.
            
        Returns:
            str: A string of today's weather data, formatted as '"column_name": value'.
        """
        # Get the coordinates
        coordinates = self.get_coordinates(location_name)

        # Get the dates
        yesterday, today = self.data_obj.get_yesterday_dates()

        # Get the weather and process
        data = self.data_obj.get_weather_data(coordinates, start_date=yesterday, end_date=today, current=1)
        data = self.data_obj.process_weather_data(data)
        data.rename(columns={"datetime": "datetime"}, inplace=True)
        data.rename(columns={"period": "datetime"}, inplace=True)
        
        # Convert to string
        row = data.iloc[0]  # Extract the single row
        data = ", ".join([f'"{col}": {row[col]}' for col in data.columns])

        return data

    def get_weather_information_tomorrow(self, location_name: str) -> str:
        """
        Returns tomorrow's weather information for a given location in a formatted string.
        
        Args:
            location_name (str): The name of the location to retrieve weather data for.
            
        Returns:
            str: A string of tomorrow's weather data, formatted as '"column_name": value'.
        """
        # Get the coordinates
        coordinates = self.get_coordinates(location_name)

        # Get the dates
        yesterday, today = self.data_obj.get_yesterday_dates()

        # Get the weather and process
        data = self.data_obj.get_weather_data(coordinates, start_date=yesterday, end_date=today, current=2)
        data = self.data_obj.process_weather_data(data)
        data.rename(columns={"datetime": "datetime"}, inplace=True)
        data.rename(columns={"period": "datetime"}, inplace=True)
        
        # Convert to string
        row = data.iloc[0]  # Extract the single row
        data = ", ".join([f'"{col}": {row[col]}' for col in data.columns])

        return data
    
    def get_ticker_name(self, company_name):
        pass

    def get_recent_news(self, company_name):
        pass

    def get_financial_statements(self, location_name: str) -> str:
        """
        Purpose:
        This function retrieves the financial balance sheets for companies in the closest ISO region 
        based on the provided location name. It fetches each company's balance sheet data and returns 
        them in a formatted string where each entry consists of the company name and its balance sheet.

        Args:
        location_name (str): The name of the location for which the closest ISO region is determined.

        Returns:
        str: A formatted string containing company names and their respective balance sheets in the 
            format "CompanyName:BalanceSheetData, CompanyName2:BalanceSheetData2, ..."
        """
        
        # Get the coordinates for the location name
        coordinates = self.get_coordinates(location_name)

        # Get the list of companies in the closest ISO region
        companies = find_closest_iso_region(coordinates)

        # Initialize an empty string to store the result
        result = ""

        # Iterate over each company in the list
        for company_entry in companies:
            # Extract company name and ticker
            company_name, ticker = company_entry.split(":")
            
            # Handle tickers with "." (split and keep the main part)
            ticker = ticker.split(".")[0]
            
            # Fetch the company's balance sheet using yfinance
            company = yf.Ticker(ticker)
            balance_sheet = company.balance_sheet

            # Check if balance sheet data is available
            if balance_sheet is not None:
                # Keep only the latest 3 columns and drop rows with NaN values
                if balance_sheet.shape[1] > 3:
                    balance_sheet = balance_sheet.iloc[:, :3]
                balance_sheet = balance_sheet.dropna(how="any")
                
                # Convert the balance sheet to a string format
                balance_sheet_str = balance_sheet.to_string()
            else:
                # If no balance sheet data is available, append a default message
                balance_sheet_str = "No data available"

            # Append the company name and balance sheet to the result string
            result += f"{company_name}:{balance_sheet_str},\n"

        return result

    def get_historical_stock_details(self, location_name:str) -> str:
        """
        Fetches the historical stock details (Close and Volume) for all companies 
        in the closest ISO region to the given location. Returns the stock history 
        for each company in the format 'company_name: historical_data' as a string.

        Args:
            location_name (str): Name of the location to find the closest ISO region.
            ticker (str): Ticker of a company for which the stock history is requested.

        Returns:
            str: A string containing the historical stock data for each company in the closest ISO region.
        """
        # Get the coordinates for the location name
        coordinates = self.get_coordinates(location_name)

        # Get the list of companies in the closest ISO region
        companies = find_closest_iso_region(coordinates)

        # Initialize a list to store the results
        result = []

        # Iterate over the companies
        for company in companies:
            # Assuming the company name follows the format "Company Name (TICKER)"
            company_ticker = company.split(":")[-1]
            
            if "." in company_ticker:
                company_ticker = company_ticker.split(".")[0]

            # Get the stock history for the ticker
            stock = yf.Ticker(company_ticker)
            df = stock.history(period="1y")
            
            # Clean the DataFrame to keep only "Close" and "Volume"
            df = df[["Close", "Volume"]]
            df.index = [str(x).split()[0] for x in list(df.index)]
            df.index.rename("Date", inplace=True)
            
            # Convert the DataFrame to string and append to result
            result.append(f"{company}: {df.to_string()}")

        # Join all the results and return as a single string
        return "\n\n".join(result)
    
    def get_stock_companies(self, location_name):
        coordinates = self.get_coordinates(location_name)
        companies = find_closest_iso_region(coordinates)
        return str(companies)
    
    # --------------------------------------------
    def init_tools(self):
        predict_energy_demand_tool = FunctionTool.from_defaults(fn=self.predict_energy_demand, 
                                                                name="predict_energy_demand", 
                                                                description="use when you want to know the energy demand for the given location. Function internally takes tomorrow's date and predicts")
        
        get_weather_information_today_tool = FunctionTool.from_defaults(fn=self.get_weather_information_today, 
                                                                        name="get_weather_information_today", 
                                                                        description="use when you want to get today's the weather information for a given name of a location")
        
        get_weather_information_tomorrow_tool = FunctionTool.from_defaults(fn=self.get_weather_information_tomorrow, 
                                                                           name="get_weather_information_tomorrow", 
                                                                           description="use when you want to get tomorrow's the weather information for a given name of a location")
        
        get_financial_statements_tool = FunctionTool.from_defaults(fn=self.get_financial_statements, 
                                                                   name="get_financial_statements_tool", 
                                                                   description="Use this to get financial statement of companies located near the location which is the input. With the help of this data company's historic performance can be evaluated, You should input the location to it")

        get_historical_stock_details_tool = FunctionTool.from_defaults(fn=self.get_historical_stock_details, 
                                                                       name="get_historical_stock_details_tool", 
                                                                       description="Use when you are asked to evaluate or analyze stocks of all companies located near the location which is the input. This will output historic share price data. You should input the location to it")
        query_tool = FunctionTool.from_defaults(fn=self.query, 
                                                name="query_tool", 
                                                description="Use this tool for any greeting and hello. Answer the question using your system prompt. The input is the query itself")
        
        self.tools = [query_tool, 
                      predict_energy_demand_tool, 
                      get_weather_information_today_tool, 
                      get_weather_information_tomorrow_tool, 
                      get_financial_statements_tool,
                      get_historical_stock_details_tool]
        
        logger.info("Added tools")

    def init_agent(self):
        self.agent = ReActAgent.from_tools(self.tools, 
                                           llm=self.llm, 
                                           system_prompt=self.system_prompt,
                                           verbose=True, 
                                           max_iterations=20)
        
        
        logger.info("Init React Agent")
    
    # --------------------------------------------
    def ingest_data(self, path):
        documents = self.read_documents(path)

        _, storage_context = self.get_storage_context()

        self.index = VectorStoreIndex.from_documents(documents, 
                                                     storage_context=storage_context, 
                                                     llm=self.llm, 
                                                     embed_model=self.embed_model)
        self.init_query_engine()
        logger.info("Ingested data successfully")

    def query(self, question, api=1):
        """Returns the answer to the question asked"""

        if api==1:
            answer = self.llm_model.generate_content(question)
            return answer.text
        else:
            question = question

            self.messages.append(ChatMessage(role="user", content=question))
            response = self.llm.chat(self.messages)
            
            return str(response)
    
    def query_custom(self, question, prompt, api=1):
        """Returns the answer to the question asked given prompt"""

        if api==1:
            prompt_question = (
                f"{prompt} \n"
                f"QUESTION: '{question}'\n"
                f"ANSWER:"
                )
            answer = self.llm_model.generate_content(prompt_question)
            return answer.text
        else:
            question = question

            messages = [ChatMessage(
                    role="system", content=prompt
                ), ChatMessage(role="user", content=question)]
            response = self.llm.chat(messages)
            
            return str(response)
        
    def query_agent(self, question):
        context = """"""
        question = question + "|" + context

        response = self.agent.chat(question)

        return str(response)
    
    def query_agent_v2(self, question):
        logger.info("Parsing input string")
        # extract city name
        response = self.query_custom(question, self.name_extraction_prompt)
        location_name = response.split(":")[-1].strip()
        logger.info(response)

        if "none" not in location_name:
            logger.info("Extracting weather, demand and stock data")
            weather_data = self.get_weather_information_today(location_name)
            predicted_demand = self.predict_energy_demand(location_name)
            stock_companies = self.get_stock_companies(location_name)
            stocks_details = self.get_historical_stock_details(location_name)
            balance_sheet_stock_details = self.get_financial_statements(location_name)

            query = question + f""" | 
            Use the below data
            Location: {location_name}
            Current Weather Data: {weather_data}
            Predicted Energy Demand: {predicted_demand}
            
            Electric Companies in {location_name}: {', '.join(stock_companies)}
            """

            # Query the LLM with the constructed prompt
            logger.info("Querying LLM")
            query_prompt = """
                            You are a weather and energy demand predictor and report generator bot.
                            Display a short paragraph on each of the above (the weather, the demand, companies effected in the area, stock trend of the company in the last few days).
                            Is the energy demand significantly high?
                            """
            
            response = self.query_custom(query, query_prompt)
            logger.info(response)
            return str(response)

        else:
            response = self.query_custom(question, "You are a redirector bot. Redirect user to ask question about energy demand prediction for certain location or stock market changes based on energy demand forcast")
            return str(response)

# rag = RAG()
# response = rag.query_agent_v2("How will tomorrow's weather effect the energy demand for Boston?")
# print(response)
