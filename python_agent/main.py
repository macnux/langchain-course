from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

load_dotenv()

def extract_code(text: str) -> str:
    """Extract Python code from LLM response."""
    # Remove think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>', '', text)
    
    # Try to find code in various formats
    patterns = [
        r'```python\n?(.*?)```',
        r'```\n?(.*?)```',
        r'Action Input:\s*(.+?)(?:\n\s*(?:Observation|Final Answer|Thought|$))',
        r'"code"\s*:\s*"(.*?)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Unescape if from JSON
            code = code.replace('\\n', '\n').replace('\\"', '"')
            # Fix truncated parentheses
            open_p = code.count('(') - code.count(')')
            open_b = code.count('[') - code.count(']')
            code += ')' * max(0, open_p)
            code += ']' * max(0, open_b)
            return code
    return None


def solve_task(task: str):
    """Solve a task using LLM to generate Python code and execute it."""
    
    llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        model="openai/gpt-oss-20b",
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Python expert. Solve the given task by writing Python code.

RULES:
1. Put your code inside a ```python code block
2. Your code MUST print the final answer
3. Do NOT use <think> tags
4. Do NOT explain - just provide the code

Example format:
```python
# your code here
print(result)
```"""),
        ("human", "{task}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    print(f"Task: {task}")
    print("=" * 50)
    print("Generating code...\n")
    
    response = chain.invoke({"task": task})
    print(f"LLM Response:\n{response}\n")
    print("=" * 50)
    
    code = extract_code(response)
    if code:
        print(f"Extracted Code:\n{code}\n")
        print("=" * 50)
        print("Execution Output:\n")
        try:
            exec(code, {"__builtins__": __builtins__})
        except Exception as e:
            print(f"Execution Error: {e}")
    else:
        print("Could not extract code from response")


def main():
    print("\n--- Python Code Generator & Executor ---\n")
    
    # Test tasks
    tasks = [
        """
 
         """
         
    ]
    
    for task in tasks:
        solve_task(task)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()