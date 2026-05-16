from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import networkx as nx
import os
import json
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import chromadb
import time
from rich.console import Console

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



console = Console()

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def pdf_to_list(path:str)->list[dict]:
    loader = PyPDFLoader(path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(pages)
    list_of_dicts = [{"id": f"doc_{i}", "text": chunk.page_content} for i, chunk in enumerate(chunks)]
    return list_of_dicts

def create_vector_store():
    """Creates a vector store in Qdrant."""

    chroma_client = chromadb.Client()
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.create_collection(name="Hybrid_RAG",embedding_function=embedding_function)
    return collection


def build_graph():
        """Returns an empty directed graph.
        Nodes = entities(concept,product,policies,people)
        Edges = relationship between them"""
        graph = nx.DiGraph()
        return graph
        

def pdf_to_str(path:str)->str:
    """Converts a PDF file to a string."""
    loader = PyPDFLoader(path)
    pages = loader.load()
    text=""
    for page in pages:
        text += page.page_content
    return text


def entities_and_relationships(text:str)->list[str]:
    """Splits the documents into Gemini and  ask it to extract (entites,relationships,targets).
    triples as json 
    These tripkes becomes edges in the knowledge graph"""
    
    prompt = f""" You are a text extraction expert, extract the key entities and relationships
    from the text below 
    Return only a JSON array . No explanation,no markdown,no code fences.
    
    Each item must have exactly These keys: 
    "entity": "the concept/product/policy/person name in text",
    "relation": "the relationship between entity and related_entity"
    "target": "the object (e.g. "Priority  Support)"
    
    text:
    {text}
    
    JSON Array:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )

    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    

    try:
        triples = json.loads(raw)
        return triples if isinstance(triples,list) else []
    except Exception as e:
        console.print(f"[RED]Failed to parse JSON: {e}[/RED]")
        console.print(f"[RED]Raw output: {raw}[/RED]")
        return []
    


def ingest_documents(documents:list[dict],collection,graph:nx.DiGraph):
    """Ingests the documents into the vector store and the knowledge graph
    Each document  is a dict with :
    "id":"unique identifier"
    "text": "content of the document"

    chromadb recieves the raw text as a chunk and the knowledge recieves extracted (entitiy,relationship
    ,target) triples.
    """
    console.print(f"[bold blue]Starting ingestion of {len(documents)} documents...[/bold blue]")

    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]

        collection.add(
            documents=[text],
            ids=[doc_id],
        )

        triples = entities_and_relationships(text)
        console.print(f"[green]Extracted {len(triples)} triples from {doc_id}[/green]")

        for triple in triples:
            entity = triple.get("entity").strip()
            relation = triple.get("relation").strip()
            target = triple.get("target").strip()

            if entity and relation and target:
                if not graph.has_node(entity):
                    graph.add_node(entity)
                if not graph.has_node(target):
                    graph.add_node(target)
                graph.add_edge(entity,target,relation=relation,source_doc=doc_id)

        console.print(f"[green]Added {doc_id} to graph[/green] {len(triples)} triples added.")
        time.sleep(15) # Wait 15s to stay under 5 RPM limit (60/5=12)


    console.print(f"[bold blue]Ingestion complete![/bold blue]"
    f"vector store has {collection.count()} documents"  
    f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                
        
    return 

def vector_retriever(query:str,collection,top_k:int=3)->list[dict]:
    """Retrieves documents from the vector store based on the query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    retrieved = []
    
    # ChromaDB returns 'documents' as a list of lists
    if "documents" in results and results["documents"]:
        for doc in results["documents"][0]:
            retrieved.append({
                "text": doc,
            })

    return retrieved


def identify_entities(query)->list[str]:
    """Ask gemini to identify which entities nodes to look up in the 
    graph database
    returns a list of entity name strings
    """

    prompt = f""" Identify the key entities (nouns,concepts,products ,policies) in the query.
    Return the names of all nodes that are relevant to the query.
    Retruns only Json array of strings ,No explanation,No markdown .

    Query:{query}
    JSON Array:
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )

    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    
    try:
        entities = json.loads(raw)
        return [str(e) for e in entities] if isinstance(entities,list) else []
    except Exception as e:
        console.print(f"[RED]Failed to parse JSON: {e}[/RED]")
        console.print(f"[RED]Raw output: {raw}[/RED]")
        return []


def graph_retriever(query:str,graph:nx.DiGraph)->list[str]:
    """
    Identify entities in the query ,finds them in the graph 
    then collects all direct neighbours and edge relationships
    Return a list of plain-English relationship strings.
    "Premium Support"--[is a]->"Enterprise Support"
    """
    entities = identify_entities(query)
    console.print(f"[bold yellow]Identified {len(entities)} entities: {entities}[/bold yellow]")
    
    # Collect related nodes and edges
    related_context = []
    for entity in entities:
        if graph.has_node(entity):
            for neighbor in graph.neighbors(entity):
                edge_data = graph.get_edge_data(entity, neighbor)
                if edge_data:
                    relation = edge_data.get('relation', 'related to')
                    context = f"{entity} --[{relation}]--> {neighbor}"
                    related_context.append(context)
                    
                    # Add reverse relationship for completeness
                    context_reverse = f"{neighbor} --[{relation}]--> {entity}"
                    related_context.append(context_reverse)
    
    console.print(f"[bold green]Found {len(related_context)} relationships in graph.[/bold green]")
    return related_context


def merge_context(vector_docs:list[str],graph_rels:list[str],query:str)->str:
    """
    Merges the vector search results and the graph-based relationships 
    into a single, highly structured context prompt for the LLM.
    Uses Markdown and XML-like tags to clearly separate different types of context.
    """
    
    parts = []
    parts.append("Here is the context retrieved to help answer the user's query.\n")
    
    # Add vector search results (documents)
    parts.append("<vector_context>")
    if vector_docs:
        parts.append("The following excerpts are from relevant documents:")
        for i, doc in enumerate(vector_docs, 1):
            parts.append(f"[Document {i}]: {doc.strip()}")
    else:
        parts.append("No relevant document excerpts found.")
    parts.append("</vector_context>\n")
    
    # Add graph relationships
    parts.append("<knowledge_graph_context>")
    if graph_rels:
        parts.append("The following factual relationships were extracted from the knowledge graph:")
        for rel in graph_rels:
            parts.append(f"- {rel}")
    else:
        parts.append("No relevant knowledge graph relationships found.")
    parts.append("</knowledge_graph_context>\n")
    
    # Add instructions and original query
    parts.append("<instructions>")
    parts.append("Please synthesize the above information from both the vector context and the knowledge graph to answer the following query.")
    parts.append(f"Query: {query}")
    parts.append("</instructions>")
    
    return "\n".join(parts)


def generate_final_answer(query:str,context:str)->str:
    """
    Generates the final answer using the LLM with the provided context.
    also explain the answer in bullet points with clear headings.
    and be according to the context provided.
    if not in context say that : No Information in context.
    """
    
    prompt = f"""Answer the following question based on the context provided.
    also explain the answer in bullet points with clear headings.
    and be according to the context provided.
    if not in context say that : No Information in context.

    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )
    
    return response.text.strip()
if __name__ == "__main__":
    sample_docs = pdf_to_list("D:/Projects/hybrid-rag/samples/2506.18027v3.pdf")

    console.print("[bold green]Initializing Vector Store and Graph...[/bold green]")
    v_store = create_vector_store()
    kg_graph = build_graph()

    # Limiting to 5 documents for debugging to avoid hitting rate limits
    ingest_documents(sample_docs[:3], v_store, kg_graph)

    
    # console.print(f"\n[bold cyan]Querying: {test_query}[/bold cyan]")


    # merged_ctx = merge_context(v_texts, g_results, test_query)
    # console.print("\n[bold]Merged Context:[/bold]")
    # console.print(merged_ctx)
    while True:
        test_query =input("\nQuery : ")
        v_results = vector_retriever(test_query, v_store)
        v_texts = [res["text"] for res in v_results]
        g_results = graph_retriever(test_query, kg_graph)

        merged_ctx = merge_context(v_texts, g_results, test_query)

        final_answer = generate_final_answer(test_query, merged_ctx)
        console.print(f"\n[bold green]Final Answer:[/bold green]\n{final_answer}")