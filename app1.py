import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from util.constants import CUSTOM_SUFFIX
from entities import Input
from llm import LLm_Init
llm_obj = LLm_Init()
import environ
env = environ.Env()
environ.Env.read_env()
# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}",
)

app = FastAPI()

app.messages = []

memory = ConversationBufferMemory(memory_key="history", input_key="input")

@app.post("/")
async def root(inp: Input):
    print(inp)

    if inp.resetFlag == 1:
        app.messages = []
    
    app.messages.append(HumanMessage(content=inp.humanMessage))
    
    agent_executor = create_sql_agent(llm_obj.llm, db=db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                      input_variables=["input", "agent_scratchpad", "history"], suffix=CUSTOM_SUFFIX,
                                      agent_executor_kwargs={"memory": memory}, verbose=True)

    response = agent_executor.invoke(app.messages)

    print(response)

    app.messages.append(AIMessage(content=response["output"]))
    return {"message": response["output"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5502)