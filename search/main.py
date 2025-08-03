from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_tool, save_tool

load_dotenv()


##prompt template, simple python class that defines the structure of the output
class ResearcherResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    response: str

##agent will being with some type of LLM
llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearcherResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful research assistant that will help generate a research paper. 
         Answer the user query and use necessary tools to gather information.
         Give the output in this format and provide no other text \n{format_instructions}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) #we are partially gonna fill in the prompt by passing the format instructions

tools = [search_tool, wikipedia_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt = prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your research query: ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response["output"])
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}", "Raw response was:", raw_response)