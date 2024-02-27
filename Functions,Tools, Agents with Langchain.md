#### OpenAI Function Calling
1. LLM can decide what functions to use, based on descriptions defined in functions;
2. when LLM get the arguments of the function, results can only obtained by invoking functions;
3. We can change the 'function_call' to 'none','auto' or force to use (function name)
```python
# define a function
# this can be backend API or external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)
# the descriptions are for LLM to decide whether to use the function
function = [
  {
    "name":"get_current_weather",
    "description": "get the current weather in a given location",
    "parameters" : {
                    "type","object",
                    "properties": {
                                  "location": {"type": "string",
                                      "description": "The city and state, e.g. San Francisco, CA",},
                                  "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]
                                  },
                    "required": ["location"],
                    },}]
# related
messages_1 = [{"role":"user", "content":"What's the weather like in Boston?"}]
# unrelated
messages_2 = [{"role":"user", "content":"What's the weather like in Boston?"}]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call='auto' # 'none', {"name": "get_current_weather"}
)
# get which/whether a tool is used
response_message = response["choices"][0]["message"]
args = json.loads(response_message["function_call"]["arguments"])
observation = get_current_weather(args)
# add user query, observation to message to get final results
messages_1.append({"role":"function","name":"get_current_weather","content":observation})
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
# assistant output
print(response["choices"][0]["message"]["content"])
```
#### LangChain Expression Language (LCEL)
<img width="447" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/8e124977-077b-4043-9327-c0841f1bb21b">
Chain = prompt | llm | OutputParser
<img width="460" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/5bfb65b7-1094-488c-b436-f2fd6aeb9b11">
LCEL advantage (Langsmith for logging and debugging)
<img width="334" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/5da72f6a-eabd-453e-be43-5ff71f4c4c9c">




