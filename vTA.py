import constants
import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = constants.APIKEY
model = "gpt-3.5-turbo"

chroma_db = "./chroma_db"
data_dir = "./data"

question = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if os.path.exists(chroma_db):
  print(f"{chroma_db} found, reusing...\n")
  vectorstore = Chroma(persist_directory=chroma_db, embedding_function=OpenAIEmbeddings())
else:
  loader = DirectoryLoader(data_dir)
  data = loader.load()
  print(f"data loaded from {data_dir}\n")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
  all_splits = text_splitter.split_documents(data)
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=chroma_db)
  vectorstore.persist()

llm = ChatOpenAI(model_name=model, temperature=0.1)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)
#qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(search_kwargs={"k": 1}), memory=memory, return_source_documents=True, verbose=False)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, return_source_documents=True, verbose=False)

while True:
  if not question:
    question = input("prompt: ")
  if question in ['quit', 'q', 'exit']:
    sys.exit()

  result = qa_chain({"question": question})

  answer = result["answer"]
  sources = set()

  print(answer)
  for source in result["source_documents"]:
    sources.add(source.metadata["source"])
  print("Sources:", *sources, sep='\n')

  question = None
