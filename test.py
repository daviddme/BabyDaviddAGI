from babydaviddagi import BabyDaviddAGI

bda = BabyDaviddAGI(
    openai_api_key="",
    pinecone_api_key="",
    pinecone_environment="",
    pinecone_table_name=""
)
idea = bda.brainstorm_agent("RSI-MACD Crossover with MACD Above 20")
print(idea)
print("="*20)
code = bda.coder_agent(idea)
print(code)
print("="*20)
print(bda.backtester_agent(code))
print("="*20)
print(bda.optimizer_agent())

# with open("./backtest.py", "r") as f:
#     bda.code = f.read()
#     print(bda.optimizer_agent())