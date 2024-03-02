Outline
1) SimpleSequentialChain: single input/output
2) SequentialChain: multiple inputs/outputs
3) RouterChain: Have multiple chains, each of them is specialized in one type of input; so first based on the input type, decide which chain to enter

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain,SequentialChain

llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")
# LLMChain
prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain = LLMChain(llm=llm, prompt = prompt)
product = "Queen Size Sheet Set"
response = chain.run(product) # dict 

# SimpleSequential Chain: input -> output_1/input_2 ->...-> output_n (response)
# Ex. input(product) -> output_1/input_2 (company_name) -> output_2/response(description)
llm = ChatOpenAI(temperature=0.9, model=llm_model)
first_prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="name")
second_prompt = ChatPromptTemplate.from_template("Write a 20 words description for the following company:{company_name}")
chain_two = LLMChain(llm=llm, prompt= second_prompt, output_key="description")
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],verbose=True)
response = overall_simple_chain.run(product) #str
```
Sequential Chian
```python
# SequentialChain: Multiple input/output
# Ex: Review -> output_1_1/input_2_1 (English_review); output_1_2/input_3_2(Language) -> output_2_1/input_3_1(summary) -> Follow-up in certain language (response)
llm = ChatOpenAI(temperature=0.9, model=llm_model)
# newline character for ""\n\n
first_prompt = ChatPromptTemplate.from_template("Translate the following review to english:\n\n{Review}")
second_prompt = ChatPromptTemplate.from_template("Can you summarize the following review in 1 sentence:""\n\n{English_Review}")
third_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
fourth_prompt = ChatPromptTemplate.from_template("Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}")
chain_one = LLMChain(llm=llm, prompt = first_prompt, output_key = "English_Review")
chain_two = LLMChain(llm=llm, prompt = second_prompt, output_key = "summary")
chain_three = LLMChain(llm=llm, prompt = third_prompt, output_key = "language")
chain_four = LLMChain(llm=llm, prompt = fourth_prompt, output_key = "followup_message")
# overall chain: input = ["Review"], output = ["English_Review", "summary","followup_message"]
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=False)
review = "..."
response = overall_chain.run(review) #dict, can extract any input, output                                           
```
Router Chain
1) Create different destination chain using multiple well-designed prompt + default chain
2) Let LLM decide which chain to use
``````python
physics_template = """You are a very smart physics professor. You are great at answering questions about physics in a concise\
and easy to understand manner. When you don't know the answer to a question you admit that you don't know.
Here is a question:
{input}"""

math_template = """You are a very good mathematician. You are great at answering math questions. You are so good because \
you are able to break down hard problems into their component parts, answer the component parts, and then put them together\
to answer the broader question.
Here is a question:
{input}"""

history_template = """..."""
computerscience_template = """..."""

prompt_info = [
  {"name": "physics", "description":"Good for answering questions about physics", "prompt_template": physics_template},
  {"name": "math", "description":"Good for answering questions about math", "prompt_template": math_template},
  {"name": "history", "description":"Good for answering questions about history", "prompt_template": history_template},
  {"name": "computerscience", "description":"Good for answering questions about computer science", "prompt_template": computerscience_template}
]
# RouterOutputParser decide which chain to use, and what's the output of that chain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser 
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model=llm_model)
# Create destination_chains for LLM to choose, so as to solve problems in different fields
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
# default_chain for problems not included into destination chain
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also \
revise the original input if you think that revising it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the 
```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=False)
response = chain.run("..") #str
``````
