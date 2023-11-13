from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.llms import Ollama, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def main(q):

    loader = TextLoader("isa-14.ttl")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)

    # embeddings = OllamaEmbeddings()
    # docsearch = Chroma.from_documents(texts, embeddings)

    # ollama = Ollama(
    #     model="llama2",
    #     verbose=True,
    #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    # )

    # qa = RetrievalQA.from_chain_type(llm=ollama, chain_type="stuff", retriever=docsearch.as_retriever())

    query = q
    qa.run(query)

if __name__ == "__main__":
    q = input("Input question:")
    main(q)