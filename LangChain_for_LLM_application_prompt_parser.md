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
3. OutputParser (define the type of LLM's output)
<img width="749" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/6f6e858b-3be3-449d-9cb8-6584b6da3a24">

we want to get dict output directly

```python
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name = "gift", description = "Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
delivery_days_schema = ResponseSchema(name = "delivery_days", description = "How many days did it take for the product to arrive? If this information is not found, \
                                      output -1.")
price_value_schema = ResponseSchema(name = "price_value", description = "Extract any sentences about the value or prive, and output them in a comma seperated Python list.")
response_schema = [gift_schema, delivery_days_schema, price_value_schema]

# after define schema for each key, can get format instruction directly (added to context)
output_parser = StructuredOutputParser.from_response_schema(response_schema)
format_instruction = output_parser.get_format_instructions()

# After getting response, use outputParser to extract information in JSON format
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(template=review_template_2)
messages = prompt.format_messages(text=response, format_instructions=format_instructions)
response = chat(messages) #JSON format
output_dict = output_parser.parse(response.content)
gift = output_dict.get('gift')
delivery_days = output_dict.get('delivery_days')
price_value = output_dict.get('price_value')
```

