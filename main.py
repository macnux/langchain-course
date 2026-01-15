from dotenv import load_dotenv
load_dotenv()
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

tools = [TavilySearch()]
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", model="openai/gpt-oss-20b", temperature=0)
client = Client()
react_prompt = client.pull_prompt("hwchase17/react")
agent = create_agent(model=llm, tools=tools, system_prompt=react_prompt.template)

def main():
    result = agent.invoke({"messages": [HumanMessage(content="what is today date and current weather in Egypt in damietta?")]})
    messages = result['messages']
    final_response = messages[-1].content
    print(final_response)

if __name__ == "__main__":
    main()
