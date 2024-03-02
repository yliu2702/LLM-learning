### Models, prompts and parser
1. Simple Prompt + LLM composition: PromptTemplate/ChatPromptTemplate -> LLM/ChatModel -> OutputParser
2. We can feed the messages (template_str -> PromptTemplate -> [input_variable + PromptTemplate] Messages) directly into model; 

```python
# set up
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
#
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
template_string = """Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"""
prompt_template = ChatPromptTemplate.from_template(template_string)
# message
message = prompt_template.format_messages(style = , text = ) #type is list; message[0] is langchain.schema.HumanMessage
response = chat(messages)


```
