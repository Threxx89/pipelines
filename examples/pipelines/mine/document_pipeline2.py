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
        self.name = "File Search RAG Pipeline"
        self.documents = None
        self.index = None
        self.summary_index = None
        self.vector_index = None
        self.retriver = None
#E:\\t5g-dev\\ProdApiJava2.0\\com.tap.t3g.fr.api.http\\docs\\
#E:\\t3g-doc-root\\test\\llama3.2:3b-instruct-fp16
        self.valves = self.Valves(
            **{
                "BASE_FILE_PATH": os.getenv("BASE_FILE_PATH", "E:\\t3g-doc-root\\test\\"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "deepseek-r1:32b-qwen-distill-q4_K_M"),
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
        from llama_index.core.node_parser import LangchainNodeParser
        from llama_index.core import SimpleDirectoryReader, StorageContext,PromptHelper
        from llama_index.core import VectorStoreIndex
        from sqlalchemy import make_url
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
        import tiktoken
        from llama_index.core.node_parser import (
            SentenceSplitter,
            SemanticSplitterNodeParser,
        )

        Settings.embed_model = OllamaEmbedding(
            temperature=0.7,
            max_tokens=2048,
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            temperature=0.7,
            max_tokens=2048,
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        Settings.text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=20,
        paragraph_separator="\n\n\n",
        secondary_chunking_regex="[^,.;]+[,.;]?",
        tokenizer= tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        )


        #loading from file
        
        reader = SimpleDirectoryReader(self.valves.BASE_FILE_PATH,recursive=True)
        self.documents  = reader.load_data()

        nodes = Settings.text_splitter.get_nodes_from_documents(self.documents , show_progress=True)

        # docstore = SimpleDocumentStore()
        # docstore.add_documents(nodes)
        # storage_context = StorageContext.from_defaults(docstore=docstore)


        chroma_client = chromadb.PersistentClient("./chroma4.db")
        collection = chroma_client.get_or_create_collection(name="Documents")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.vector_index = VectorStoreIndex(
                                nodes=nodes, 
                                storage_context=storage_context, 
                                show_progress=True, 
                                embed_model=Settings.embed_model
                            )
    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        from llama_index.core import VectorStoreIndex, get_response_synthesizer
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.postprocessor import SimilarityPostprocessor
        from llama_index.llms.ollama import Ollama        
        from llama_index.core.response_synthesizers import ResponseMode

        llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            temperature=0.7,
            max_tokens=2048
        )
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=30,
        )

        # # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.ACCUMULATE)

        # # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        )

        # query
        response = self.vector_index.as_query_engine().query(user_message)
        print(response)
        return response.response
    

    