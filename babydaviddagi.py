from typing import Optional, Dict, Union, List
import logging, time, subprocess, sys, re, random, json, uuid
import openai
import pinecone




class BabyDaviddAGI:
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        openai_api_model: str = "gpt-3.5-turbo",
        pinecone_environment: str = "asia-southeast1-gcp",
        pinecone_table_name: str = "babydaviddagi",
        objective: str = "profitable trading strategy",
    ) -> None:
        """
        Initialize the BabyDaviddAGI instance with the provided API keys, model name, environment, table name, and objective.

        Args:
            openai_api_key (str): The API key for OpenAI.
            pinecone_api_key (str): The API key for Pinecone.
            openai_api_model (str, optional): The model name for OpenAI API. Default is "gpt-3.5-turbo".
            pinecone_environment (str, optional): The environment for Pinecone. Default is "asia-southeast1-gcp".
            pinecone_table_name (str, optional): The table name for Pinecone index. Default is "babydaviddagi".
            objective (str, optional): The objective of the trading strategy. Default is "profitable trading strategy".

        Returns:
            None
        """
        logging.basicConfig(
            format="%(asctime)s : \u001b[33m%(levelname)s\u001b[0m : \033[91m\033[1m%(message)s\u001b[0m",
            datefmt="%m/%d/%Y %I:%M:%S",
        )
        self._openai_api_key = openai_api_key
        self._pinecone_api_key = pinecone_api_key
        self._openai_api_model = openai_api_model
        self._pinecone_environment = pinecone_environment
        self._pinecone_table_name = pinecone_table_name

        openai.api_key = self._openai_api_key

        pinecone.init(
            api_key=self._pinecone_api_key, environment=self._pinecone_environment
        )
        if self._pinecone_table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self._pinecone_table_name,
                dimension=1536,
                metric="cosine",
                pod_type="p1",
            )
        self.pine_index = pinecone.Index(self._pinecone_table_name)
        self.OBJECTIVE = objective

        self.idea = ""
        self.code = ""
        self.bactest_result = ""
        self.optimizer_result = ""
        self.analyst_result: Dict[str, str] = {}

        if "gpt-4" in self._openai_api_model.lower():
            logging.warn("Be Carefully! You're using GPT-4. Monitor your costs")
            
    def start(self, user_prompt:str) -> None:
        """
        Start the BabyDaviddAGI pipeline with the provided user prompt.

        Args:
            user_prompt (str): The user prompt to initiate the pipeline.

        Returns:
            None
        """
        while True:
            self.brainstorm_agent(user_specs=user_prompt)
            self.coder_agent()
            self.backtester_agent()
            self.optimizer_agent()
            if self.analyst_agent()["rate"] >= 5:
                break 

    def brainstorm_agent(self, user_specs: str) -> str:
        """
        Generate brainstorming ideas for a given user specification using OpenAI's GPT model.

        Args:
            user_specs (str): User specification for brainstorming ideas.

        Returns:
            str: Generated brainstorming ideas.

        """
        last_datas = self.context_agent(query=user_specs, n=5)

        prompt = f"""
        You are a task creation AI responsible for brainstorming idea for {self.OBJECTIVE} based on {user_specs},
        The last generated ideas: {data['idea'] for data in last_datas}.
        Results of that ideas: {data['result'] for data in last_datas}. 
        Don't overlap with last ideas. Idea dont exceed 150 word"""
        self.idea = self.__openai_call(prompt)
        return self.idea

    def coder_agent(self) -> str:
        """
        Generate Python code using brainstorm idea and OpenAI's GPT model.

        Returns:
            str: Generated Python Code.

        """
        prompt = f"""
        You are advance level Python developer who performs one task based on the following objective: {self.OBJECTIVE}.
Your task: {self.idea}.
I created template python code, i will provide to you. Create indicator variables using talib.
Here is the my code. Edit this code. I wrote comments to change. Define indicators like i provide and edit if elif conditions but dont touch other codes. 
Also don't use complicated conditions and don't create a new class.
Return just python code.
# you can add new import from backtesting but you can't delete any import from here
import talib
from backtesting import Strategy, Backtest
from backtesting.test import GOOG

class SmaCross(Strategy):
    # Create indicator variables here which you will give parameter in self.I function.
    n1 = 30
    n2 = 50
    def init(self):
        # edit here: define every indicator here as a class variable. Don't forget every indicator have to be in self.I function.
        # also dont give name and dont create indicator variable in init function all variables have to be defined before this function
        self.sma1 = self.I(talib.SMA, self.data.Close, self.n1)
        self.sma2 = self.I(talib.RSI, self.data.Close, self.n2)
    
    def next(self):
        # edit if conditon: check buy conditions example provided here
        #  buy the asset
        if self.sma1 > self.sma2:
            self.buy()

        # edit elif condition: check sell conditions example provided here
        # sell the asset
        elif (self.sma1 < n2):
            self.sell()
            
bt = Backtest(GOOG, SmaCross, cash=10_000, commission=.002)
stats = bt.run()

# dont change or delete after here
print("Final Equity: ", stats["Equity Final [$]"])
print("Peak Equity ", stats["Equity Peak [$]"])
print("Return Rate: ", stats["Return [%]"])
print("Buy and hold return rate: ", stats["Buy & Hold Return [%]"])
print("Sharpe Ratio: ", stats["Sharpe Ratio"])
print("Max Drawdown Rate: ", stats["Max. Drawdown [%]"])
print("Trade Count", stats["# Trades"])
        """
        self.code = self.__openai_call(prompt, temperature=0.7, max_tokens=2000)
        return self.code

    def backtester_agent(self) -> str:
        """
        Perform backtesting of a trading strategy using the code generated by the coder agent.

        Returns:
            str: Results of the backtesting.

        """
        code = 'import warnings\nwarnings.filterwarnings("ignore")\n' + self.code
        self.code = code
        with open("./backtest.py", "w") as fp:
            fp.write(code)

        try:
            output = (
                subprocess.run(
                    [sys.executable, "./backtest.py"], capture_output=True, check=True
                )
                .stdout.decode("utf-8")
                .replace("\n", " ")
                .replace("\r", "")
            )
        except subprocess.CalledProcessError as e:
            output = e.stderr
        self.bactest_result = str(output)
        return self.bactest_result

    def optimizer_agent(self):
        """
        Perform optimization of backtested results using the parameters generated by the optimizer generator.

        Returns:
            str: Results of the optimization.

        """
        params = self.__optimizer_generator()
        func_params = ", ".join([f"{item[0]}={item[1]}" for item in params.items()])

        func = f"stats = bt.optimize({func_params})"

        contents = self.code.splitlines()

        contents.insert(len(contents) - 8, func)

        with open("./backtest.py", "w") as f:
            contents = "\n".join(contents)
            f.write(contents)

        try:
            output = (
                subprocess.run(
                    [sys.executable, "./backtest.py"], capture_output=True, check=True
                )
                .stdout.decode("utf-8")
                .replace("\n", " ")
                .replace("\r", "")
            )
        except subprocess.CalledProcessError as e:
            output = e.stderr
        self.optimizer_result = str(output)
        return self.optimizer_result

    def analyst_agent(self):
        """
        Act like a data analyst, evaluate the backtest results for a trading strategy, and provide a grade from 1 to 10,
        where 1 indicates a bad result and 10 indicates a good result. Also, analyze for potential overfitting of the strategy.
        Returns the analyst's evaluation as a JSON object.

        Returns:
            dict: Analyst's evaluation in the form of a JSON object with keys 'rate' and 'summary'.
        """
        prompt = f"""
        Act like a data analyst, you are working for a large quant trading firm. 
I have created the  "{self.idea}" strategy. 
I want you to look at the results and grade them from 1 to 10, 1 being bad and 10 being good. 
My objective is to trade this on crypto markets. 
Please analyst for potential overfitting. 
Backtest results on original settings : "{self.bactest_result}" 
Backtest results on optimised results : "{self.optimizer_result}".
I want result as just json. For example {{
"rate":  # your point,
"summary": # explain why you give that point
}}
        """
        analyst_point = json.loads(self.__openai_call(prompt))

        vector = self.__get_embedding(self.idea)
        self.pine_index.upsert(
            vectors=[
                (
                    uuid.uuid4().hex,
                    vector,
                    {"idea": self.idea, "result": analyst_point["summary"]}
                )
            ],
            namespace=self.OBJECTIVE
        )
        self.analyst_result = analyst_point
        return analyst_point

    def context_agent(self, query: str, n: int):
        """
        Retrieve context from the Pine database based on a given query and number of responses.
        
        Args:
            query (str): Query to search for in the database.
            n (int): Number of responses to retrieve.
        
        Returns:
            list: List of tuples containing the idea and result metadata from the Pine database.
        """
        embedded_vectors = self.__get_embedding(query)
        query_responses = self.pine_index.query(
            embedded_vectors, top_k=n, include_metadata=True, namespace=self.OBJECTIVE
        )
        sorted_responses = sorted(
            query_responses.matches, key=lambda x: x.score, reverse=True
        )
        return [
            (str(response.metadata["idea"]), str(response.metadata["result"]))
            for response in sorted_responses
        ]

    def __openai_call(
        self,
        prompt: str,
        temperature: Optional[float] = 0.5,
        max_tokens: Optional[int] = 200,
    ) -> str:
        """
        Call the OpenAI API to generate text based on the given prompt using the configured model.
        
        Args:
            prompt (str): The prompt to generate text from.
            temperature (Optional[float]): The temperature parameter for controlling the randomness of the generated text.
                                        Defaults to 0.5.
            max_tokens (Optional[int]): The maximum number of tokens in the generated text. Defaults to 200.
        
        Returns:
            str: The generated text from the OpenAI API response.
        """
        try:
            messages = [{"role": "system", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self._openai_api_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            logging.warn(
                "OpenAI API RateLimit has been exceeded. Waiting 10 second and trying again."
            )
            time.sleep(10)
            return self.__openai_call(prompt, temperature, max_tokens)
        except Exception as e:
            logging.critical(f"Unknown error while OpenAPI Call: {e}")
            return ""

    def __get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Get the embedding vector for the given text using the specified OpenAI embedding model.
        
        Args:
            text (str): The input text to get the embedding vector for.
            model (str): The name of the OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        
        Returns:
            List[float]: The embedding vector for the given text.
        """
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0][
            "embedding"
        ]

    def __optimizer_generator(self) -> Dict:
        """
        Generate optimizer data from the code string.
        
        Returns:
            Dict: A dictionary containing optimizer data extracted from the code string.
        """
        datas = {}
        for line in self.code.splitlines():
            find = re.search(r".* = [0-9]+", line)
            if find:
                find_var = find.group(0).strip().split(" = ")
                datas[find_var[0]] = range(
                    int(int(find_var[1]) / 2),
                    int(find_var[1]) * 2,
                    random.choice([3, 5, 7]),
                )

        if datas == {}:
            count = 0
            temp_data = []
            temp = self.code.splitlines()
            for num, line in enumerate(self.code.splitlines()):
                find = re.search(r"(self.I)(.*)", line)
                if find:
                    find_var = find.group(2).split(",")[2:]
                    data = {}
                    for var in find_var:
                        var = var.replace("(", "").replace(")", "")
                        if "=" in var:
                            data[f"n{count}"] = var.split("=")[1].strip()
                        else:
                            data[f"n{count}"] = var.strip()

                        count += 1
                    temp_data.append(data)
                    temp[
                        num
                    ] = f'{temp[num].split("=", 1)[0]} = self.I{find.group(2).split(",")[0]}, {find.group(2).split(",")[1]}, {", ".join([f"self.{x[0]}" for x in data.items()])})'

            for num, line in enumerate(self.code.splitlines()):
                if "class" in line:
                    for variables in temp_data:
                        for var in variables.items():
                            temp.insert(num + 1, f"    {var[0]} = {var[1]}")

            self.code = "\n".join(temp)
            return self.__optimizer_generator()

        return datas
