### Two types of large language model
1. Base LLM : predict next word, based on training data
     
   I love (eating my mother's soup); What's the capital of France? (What's the France's largest city? ..)
2. Instruction-tuned LLM: follow the instruction

   What's the capital of France? (Paris)
3. Train Base LLM to an instruction tuned LLM
   
   1) Train a base LLM on a lot of data
   2) Finetune LLM on examples of where outputs follows input instructions
   3) Obtrain human-ratings of the quality of different LLM outputs, on criteria whether it's helpful, honest and harmless
   4) Tune LLM to increase the probability that LLM generates outputs with higer scores (using RLHF)
4. LLM understand words as composition of tokens (1-4 characters); some LLM have different token limit of input 'context' +output completion, like 'gpt3.5-turbo' ~ 4000 tokens
5. System, User, Assistant messages:
<img width="391" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/9294f0dc-09e3-4346-89b0-fc84dfa01080">
```python
{{
messages = [
{'role':'system', 
 'content':"""You are an assistant who responds\
 in the style of Dr Seuss."""},    
{'role':'user',
 'content':"""write me a very short poem \ 
 about a happy carrot"""},  
] }}
```

6. 

