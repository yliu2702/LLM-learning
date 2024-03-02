Outline
1) ConversationBufferMemory
2) ConversationWindowBufferMemory: only remember last window's infomation
3) ConversationTokenBufferMemory: only remember conversational history within certain tokens
4) ConversationSummaryMemory: LLM loads a summary of previous conversations within certain token

Creat memory -> save_context in memory -> create model -> model.predict -> check response or history

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# create memory and load model
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
# add elements in memory
# memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)
```
