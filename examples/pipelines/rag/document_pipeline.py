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
        self.retriver = None
#E:\\t5g-dev\\ProdApiJava2.0\\com.tap.t3g.fr.api.http\\docs\\
#E:\\t3g-doc-root\\test\\
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
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        from llama_index.core import SimpleDirectoryReader, StorageContext
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.postgres import PGVectorStore
        from sqlalchemy import make_url
        from langchain.document_loaders import TextLoader
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.vectorstores import Pinecone
        from langchain.document_loaders import TextLoader
        from langchain.vectorstores.pgvector import PGVector
        import psycopg2

        Settings.embed_model = OllamaEmbedding(
            temperature=0,
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        #loading from file
        
        reader = SimpleDirectoryReader(self.valves.BASE_FILE_PATH,recursive=True)
        self.documents  = reader.load_data()
        #nodes = SentenceSplitter(chunk_size=8192,chunk_overlap=160).get_nodes_from_documents(self.documents)
        #https://github.com/daveebbelaar/langchain-experiments/blob/main/pgvector/pgvector_service.py
        #parser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=16384, chunk_overlap=1024))
        #nodes = parser.get_nodes_from_documents(self.documents)


        #docstore = SimpleDocumentStore()
        #docstore.add_documents(nodes)
        #storage_context = StorageContext.from_defaults(docstore=docstore)
        
        #self.vector_index = VectorStoreIndex( nodes, storage_context=storage_context, embed_model=Settings.embed_model)

        # CONNECTION_STRING = PGVector.connection_string_from_db_params(
        #     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        #     host=os.environ.get("PGVECTOR_HOST", "localhost"),
        #     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        #     database=os.environ.get("PGVECTOR_DATABASE", "RAG"),
        #     user=os.environ.get("PGVECTOR_USER", "postgres"),
        #     password=os.environ.get("PGVECTOR_PASSWORD", "postres"),
        # )
        #COLLECTION_NAME = "The Project Gutenberg eBook of A Christmas Carol in Prose"
        # create the store
        # db = PGVector.from_documents(
        #     embedding= Settings.embed_model,
        #     documents=self.documents,
        #     collection_name=COLLECTION_NAME,
        #     connection_string=CONNECTION_STRING,
        #     pre_delete_collection=False,
        # )
        connection_string = "postgresql://postgres:postgres@localhost:5432"
        db_name = "RAG"
        # conn = psycopg2.connect(connection_string)
        # conn.autocommit = True

        url = make_url(connection_string)
        PGVectorStore.from_orm
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="paul_graham_essay",
            embed_dim=768,  # openai embedding dimension
            hnsw_kwargs={
                "hnsw_m": 32,
                "hnsw_ef_construction": 128,
                "hnsw_ef_search": 80,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #From DB Directly
        #self.vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        #Saving to DB
       # self.vector_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)
        self.vector_index  = VectorStoreIndex.from_documents(
        self.documents  , storage_context=storage_context, show_progress=True, embed_model=Settings.embed_model
        )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from llama_index.core import get_response_synthesizer
        from llama_index.core.response_synthesizers import ResponseMode
        from llama_index.llms.ollama import Ollama

        llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.TREE_SUMMARIZE
        )
        # prompt_template = """You are a expert. You need to answer the question related to software development. 
        # Given below is the context and question of the user.
        # context = {context}
        # question = {user_message}
        # """
        self.retriver = self.vector_index.as_retriever()
     
        
        #query_engine = self.vector_index.as_query_engine(streaming=True)
        #response = query_engine.query(user_message)
        engine = RAGQueryEngine(retriever=self.retriver, response_synthesizer=response_synthesizer)
        response = engine.custom_query(user_message)
        return response.response
    
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor

class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        #postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        #nodes = postprocessor.postprocess_nodes(nodes)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj
    # def process_directory(self, directory):
    #     for root, dirs, files in os.walk(directory):
    #         if len(files) > 0:
    #             try:
    #                 reader = SimpleDirectoryReader(directory)
    #                 self.documents = reader.load_data()
    #                 self.nodes += MarkdownElementNodeParser(chunk_size=1024).get_nodes_from_documents(self.documents)
    #             except:
    #                 pass
    #         for dirs_name in dirs:
    #             self.process_directory(os.path.join(directory, dirs_name))