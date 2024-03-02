### Embedding
1. Since LLMs can only inspect thousands of words at a time, we create embedding vectors to capture content/meaning
2. We create embedding vector database, which stores the embedded chunk vectors and orginal chunks of the document
  <img width="781" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/a68c8365-19cc-4605-932a-e331132e5749">
3. We index the chunks in vector database; embeded the user query and search for the most similar embeded chunks in database with the embeded query
  <img width="791" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/ecae3cd5-bb01-4b82-b5b9-c2a90cb7a7bb">
4. The returned embedded chunks / original chunks can fit in the LLM's context

Step-by-Step
```python
# load document
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)
docs = loader.load()
# embedding model, used on query and doc
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
query_embed = embeddings.embed_query("Hi my name is Harrison")
db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# can directly search unembeded user query in db for related chunks
docs = db.similarity_search("Please suggest a shirt with sunblocking")
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
llm = ChatOpenAI(temperature = 0.0, model="gpt-3.5-turbo")
query =  "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = llm.call_as_llm(f"{qdocs} Question: {}")
# Table of product_name and description
display(Markdown(response))
```
Result:
<img width="755" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/04e2c7fb-a3c0-48a0-b001-7d077f1b3637">

```python
# Another method using db
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model="gpt-3.5-turbo")
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
response = qa_stuff.run(query)
display(Markdown(response))
```
Result:
<img width="744" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/8dc6b6d3-6c95-4f8a-913e-53c2338dca6c">

Most direct
```python
loader = CSVLoader(file_path=file)
index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch,embedding=embeddings).from_loaders([loader])
query ="Please list all your shirts with sun protection in a table in markdown and summarize each one."
llm_replacement_model = OpenAI(temperature=0, model='gpt-3.5-turbo-instruct')
response = index.query(query, llm=llm_replacement_model)
display(Markdown(response))
```
<img width="739" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/eeed8891-40d6-4add-83f6-c5567e070415">

if use llm = ChatOpenAI(temperature = 0.0, model="gpt-3.5-turbo")
<img width="746" alt="image" src="https://github.com/yliu2702/LLM-learning/assets/154867456/60f8d5f8-6acb-4bcd-9294-e1322abe061f">
