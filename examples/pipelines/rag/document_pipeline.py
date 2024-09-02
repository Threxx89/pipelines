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
        self.name = "Document RAG Pipeline"
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

        
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.chunk_size = 1024

        reader = SimpleDirectoryReader(self.valves.BASE_FILE_PATH)
        documents = reader.load_data()
        self.documents = SimpleDirectoryReader("E:\\t5g-dev\\ProdApiJava2.0\\com.tap.t3g.fr.api.http\\docs\\").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        nodes = SentenceSplitter().get_nodes_from_documents(documents)
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)
        self.summary_index = SummaryIndex(nodes, storage_context=storage_context)
        self.vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        self.keyword_table_index = SimpleKeywordTableIndex(
            nodes, storage_context=storage_context
        )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.



        query_engine = self.summary_index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        query_engine = self.vector_index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        query_engine = self.keyword_table_index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
