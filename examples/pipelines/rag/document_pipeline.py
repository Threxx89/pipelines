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
from llama_index.core import  PromptTemplate
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        BASE_FILE_PATH: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str       
        LLAMAINDEX_OLLAMA_BASE_URL: str

    def __init__(self):
        self.name = "Document RAG Pipeline"
        self.documents = None
        self.index = None
        self.summary_index = None
        self.vector_index = None
        self.keyword_table_index = None

        self.valves = self.Valves(
            **{
                "BASE_FILE_PATH": os.getenv("BASE_FILE_PATH", "E:\\t3g-doc-root\\test\\"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b-instruct-q8_0"),
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text:v1.5"),
            }
        )

    async def on_startup(self):
        import os 
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import VectorStoreIndex
        from llama_index.core import Settings
        from llama_index.core.node_parser import SentenceSplitter 
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core import StorageContext
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core.node_parser import SentenceSplitter

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.chunk_size = 1024

        reader = SimpleDirectoryReader(self.valves.BASE_FILE_PATH)
        self.documents  = reader.load_data()
        nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(self.documents)
        
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)
        
        self.vector_index = VectorStoreIndex( nodes, storage_context=storage_context)


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        text_to_sql_template = PromptTemplate("Your are an expert on the provided document. You need to give a detailed response to the users question. Please provide an accurate answer based on the following question {user_message}. Using the language provide by the question.")
        #query_engine = self.vector_index.as_query_engine(streaming=True,response_mode="tree_summarize", prompt_template=text_to_sql_template)
        query_engine = self.vector_index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
