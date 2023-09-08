import constants
import os
import sys

from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredPowerPointLoader
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
#model = "gpt-4"

chroma_db = "./chroma_db"
doc_dir = "./documents"
question = None

if len(sys.argv) > 1:
  query = sys.argv[1]

documents = []
if os.path.exists(chroma_db):
  vectorstore = Chroma(persist_directory=chroma_db, embedding_function=OpenAIEmbeddings())
else:
  print(f"{chroma_db} not found, creating")
  print(f"loading documents from {doc_dir}")
  documents = []
  for file in os.listdir(doc_dir):
    if file.endswith(".pdf"):
      print(f"  pdf: {file}")
      pdf_path = doc_dir + "/" + file
      loader = PyPDFLoader(pdf_path)
      documents.extend(loader.load())
    elif file.endswith(".docx") or file.endswith(".doc"):
      print(f"  doc: {file}")
      doc_path = doc_dir + "/" + file
      loader = Docx2txtLoader(doc_path)
      documents.extend(loader.load())
    elif file.endswith(".pptx") or file.endswith(".ppt"):
      print(f"  ppt: {file}")
      ppt_path = doc_dir + "/" + file
      loader = UnstructuredPowerPointLoader(ppt_path)
      documents.extend(loader.load())
    elif file.endswith(".txt"):
      print(f"  txt: {file}")
      txt_path = doc_dir + "/" + file
      loader = TextLoader(txt_path)
      documents.extend(loader.load())

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
  all_splits = text_splitter.split_documents(documents)
  vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=chroma_db)
  vectorstore.persist()

llm = ChatOpenAI(model_name=model, temperature=0.1)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, return_source_documents=True, verbose=True)

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
