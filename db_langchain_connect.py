from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from llm import LLm_Init
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import environ
env = environ.Env()
environ.Env.read_env()
llm_obj = LLm_Init()

# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}",
)

llm = llm_obj.llm

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
answer = answer_prompt | llm | StrOutputParser()

system = """Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only."""
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{query}")]
).partial(dialect=db.dialect)
validation_chain = prompt | llm | StrOutputParser()

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                chain = create_sql_query_chain(llm, db)
                full_chain = {"query": chain} | validation_chain
                response = full_chain.invoke({"question": prompt})
                print(response)
                result = db.run(response.strip("```sql"))
                print(answer.invoke({"question": prompt, "query": response, "result": result}))
            except Exception as e:
                print(e)

get_prompt()