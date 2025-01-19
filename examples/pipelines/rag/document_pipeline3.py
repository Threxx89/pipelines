"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        BASE_FILE_PATH: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str       
        LLAMAINDEX_OLLAMA_BASE_URL: str

    def __init__(self):
        self.name = "Document RAG Pipeline V2"
        self.documents = None
        self.index = None
        self.summary_index = None
        self.vector_index = None
        self.keyword_table_index = None

        self.valves = self.Valves(
            **{
                "BASE_FILE_PATH": os.getenv("BASE_FILE_PATH", "E:\\t5g-dev\\ProdApiJava2.0\\com.tap.t3g.fr.api.http\\docs\\"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b-instruct-q8_0"),
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text:v1.5"),
            }
        )

    async def on_startup(self):
        import os 
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
        from llama_index.core import SummaryIndex
        from llama_index.core import ComposableGraph
        from llama_index.llms.openai import OpenAI
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core import StorageContext
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        import os
        from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext
        from langchain import OpenAI
        
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.chunk_size = 1024



        # Set OpenAI API key
        os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'

        # Initialize the LLM predictor
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="llama3.1:8b-instruct-q8_0"))

        # Create a service context
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        # Load documents from a directory
        documents = SimpleDirectoryReader('E:\\t3g-doc-root\\test').load_data()

        # Create a vector store index
        self.index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)




    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.



        # Build a query engine
        query_engine = self.index.as_query_engine(streaming=True)

        # Query the index
        response = query_engine.query("What is the document about?")

        return response.response_gen
