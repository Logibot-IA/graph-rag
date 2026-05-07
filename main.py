import os
import uuid
import json
import re
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from dataclasses import dataclass, field

import networkx as nx

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from rag_settings import (
    build_embeddings,
    build_llm,
    build_ragas_llm,
    configure_environment,
    extract_response_text,
    finish_usage_tracker,
    get_chroma_settings,
    run_ragas as avaliar_com_ragas,
    salvar as salvar_resultados,
    start_usage_tracker,
)

configure_environment("benchmark-graph-rag")

DOCS_DIR = os.getenv("DOCS_DIR", "./docs/")
PERSIST_DIR, CHROMA_COLLECTION_NAME = get_chroma_settings(
    "./chroma_graph_db_openai",
    "graph_collection_openai",
)

test_queries = [
    # FÁCEIS
    "O que significa ‘lógica de programação’ em palavras simples?",
    "De um jeito bem direto: o que é um algoritmo?",
    "Qual é a diferença entre constante e variável?",
    "Pra que serve o comando ‘leia’ em um algoritmo?",

    # MÉDIAS
    "O que é um comando de atribuição e por que o tipo do dado precisa ser compatível com o tipo da variável?",
    "O que são operadores aritméticos (como +, -, * e /) e pra que eles servem?",
    "Pra que servem os operadores relacionais numa expressão?",

    # DIFÍCEIS
    "O que é uma ‘expressão lógica’?",
    "Em uma repetição, o que é um contador e como ele é incrementado?",
    "Como funciona a repetição ‘repita ... até’ e o que ela garante sobre a execução do bloco?"
]


ground_truths = [
    # FÁCEIS
    "Lógica de programação é o uso correto das leis do pensamento, da ‘ordem da razão’ e de processos formais de raciocínio e simbolização na programação de computadores, com o objetivo de produzir soluções logicamente válidas e coerentes para resolver problemas.",
    "Um algoritmo é uma sequência de passos bem definidos que têm por objetivo solucionar um determinado problema.",
    "Um dado é constante quando não sofre variação durante a execução do algoritmo: seu valor permanece constante do início ao fim (e também em execuções diferentes ao longo do tempo). Já um dado é variável quando pode ser alterado em algum instante durante a execução do algoritmo, ou quando seu valor depende da execução em um certo momento ou circunstância.",
    "O comando de entrada de dados ‘leia’ é usado para que o algoritmo receba os dados de que precisa: ele tem a finalidade de atribuir o dado fornecido à variável identificada, seguindo a sintaxe leia(identificador) (por exemplo, leia(X) ou leia(A, XPTO, NOTA)).",

    # MÉDIAS
    "Um comando de atribuição permite fornecer um valor a uma variável. O tipo do dado atribuído deve ser compatível com o tipo da variável: por exemplo, só se pode atribuir um valor lógico a uma variável declarada como do tipo lógico.",
    "Operadores aritméticos são o conjunto de símbolos que representam as operações básicas da matemática (por exemplo: + para adição, - para subtração, * para multiplicação e / para divisão). Para potenciação e radiciação, o livro indica o uso das palavras-chave pot e rad.",
    "Operadores relacionais são usados para realizar comparações entre dois valores de mesmo tipo primitivo. Esses valores podem ser constantes, variáveis ou expressões aritméticas, e esses operadores são comuns na construção de equações.",

    # DIFÍCEIS
    "Uma expressão lógica é aquela cujos operadores são lógicos ou relacionais e cujos operandos são relações, variáveis ou constantes do tipo lógico.",
    "Um contador é um modo de contagem feito com a ajuda de uma variável com um valor inicial, que é incrementada a cada repetição. Incrementar significa somar um valor constante (normalmente 1) a cada repetição.",
    "A estrutura de repetição ‘repita ... até’ permite que um bloco (ou ação primitiva) seja repetido até que uma determinada condição seja verdadeira. Pela sintaxe da estrutura, o bloco é executado pelo menos uma vez, independentemente da validade inicial da condição."
]

llm = build_llm()
llm_extractor = build_llm()
embeddings = build_embeddings()
vector_store = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)


@dataclass
class EntityNode:
    id: str
    name: str
    type: str
    description: str = ""
    chunk_ids: List[str] = field(default_factory=list)


@dataclass
class RelationEdge:
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, EntityNode] = {}
        self._name_index: Dict[str, str] = {}

    def add_entity(self, entity: EntityNode) -> None:
        self.entities[entity.id] = entity
        self._name_index[entity.name.lower()] = entity.id
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
        )

    def add_relation(self, relation: RelationEdge) -> None:
        if relation.source_id in self.entities and relation.target_id in self.entities:
            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation=relation.relation_type,
                description=relation.description,
                weight=relation.weight,
            )

    def find_entity_by_name(self, name: str) -> Optional[EntityNode]:
        name_lower = name.lower()
        if name_lower in self._name_index:
            return self.entities[self._name_index[name_lower]]
        for stored_name, eid in self._name_index.items():
            if name_lower in stored_name or stored_name in name_lower:
                return self.entities[eid]
        return None

    def get_neighbors(self, entity_id: str, depth: int = 2) -> List[str]:
        visited = set()
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                nbrs = set(self.graph.successors(nid)) | set(self.graph.predecessors(nid))
                next_frontier.update(nbrs - visited)
            visited.update(frontier)
            frontier = next_frontier
        visited.update(frontier)
        return list(visited)

    def build_context(self, node_ids: List[str]) -> str:
        if not node_ids:
            return "Nenhuma entidade relevante encontrada no grafo."
        subgraph = self.graph.subgraph(node_ids)
        lines = ["### Entidades Relevantes do Grafo\n"]
        for nid in subgraph.nodes():
            entity = self.entities.get(nid)
            if entity:
                lines.append(f"- **{entity.name}** [{entity.type}]: {entity.description}")
        lines.append("\n### Relações\n")
        for u, v, data in subgraph.edges(data=True):
            u_name = self.entities.get(u, EntityNode(u, u, "")).name
            v_name = self.entities.get(v, EntityNode(v, v, "")).name
            rel = data.get("relation", "RELATES_TO")
            desc = data.get("description", "")
            lines.append(f"- {u_name} --[{rel}]--> {v_name}: {desc}")
        return "\n".join(lines)

    @property
    def stats(self) -> Dict[str, int]:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}


knowledge_graph = KnowledgeGraph()

EXTRACTION_SYSTEM = """Você é um extrator especializado de entidades e relações.
Retorne APENAS um JSON válido, sem texto adicional, sem markdown."""

EXTRACTION_TEMPLATE = """Analise o texto e extraia entidades e relações.

Retorne SOMENTE este JSON (sem blocos de código, sem explicações):
{{
  "entities": [
    {{"id": "e1", "name": "Nome da Entidade", "type": "Concept|Person|Organization|Location|Event", "description": "descrição breve"}}
  ],
  "relations": [
    {{"source": "e1", "target": "e2", "type": "RELATES_TO|IS_A|PART_OF|CAUSES|DEFINES", "description": "como se relacionam"}}
  ]
}}

Texto:
{text}"""


def extract_entities_from_chunk(chunk: Document) -> Dict:
    prompt = EXTRACTION_TEMPLATE.format(text=chunk.page_content[:1500])
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(content=prompt),
    ]
    response = llm_extractor.invoke(messages)
    raw = extract_response_text(response).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}


def ingest_chunk_into_graph(chunk: Document, extraction: Dict) -> None:
    chunk_id = chunk.metadata.get("chunk_id", str(uuid.uuid4()))
    local_id_map: Dict[str, str] = {}

    for ent in extraction.get("entities", []):
        global_id = f"{ent['name'].lower().replace(' ', '_')}_{ent['type'].lower()}"
        local_id_map[ent["id"]] = global_id
        if global_id not in knowledge_graph.entities:
            knowledge_graph.add_entity(EntityNode(
                id=global_id,
                name=ent["name"],
                type=ent.get("type", "Concept"),
                description=ent.get("description", ""),
                chunk_ids=[chunk_id],
            ))
        else:
            knowledge_graph.entities[global_id].chunk_ids.append(chunk_id)

    for rel in extraction.get("relations", []):
        src = local_id_map.get(rel["source"])
        tgt = local_id_map.get(rel["target"])
        if src and tgt:
            knowledge_graph.add_relation(RelationEdge(
                source_id=src,
                target_id=tgt,
                relation_type=rel.get("type", "RELATES_TO"),
                description=rel.get("description", ""),
            ))


@tool(response_format="content_and_artifact")
def retrieve_vector_context(query: str):
    """Recupera chunks de texto relevantes por similaridade semântica."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


@tool(response_format="content_and_artifact")
def retrieve_graph_context(query: str):
    """Recupera contexto estruturado do grafo de conhecimento: entidades e relações relevantes à query."""
    words = [w.strip('.,;:?!"') for w in query.split() if len(w) > 3]
    relevant_nodes: List[str] = []
    for word in words:
        entity = knowledge_graph.find_entity_by_name(word)
        if entity:
            neighbors = knowledge_graph.get_neighbors(entity.id, depth=2)
            relevant_nodes.extend(neighbors)
    relevant_nodes = list(set(relevant_nodes))[:40]
    graph_context = knowledge_graph.build_context(relevant_nodes)
    return graph_context, relevant_nodes


tools = [retrieve_vector_context, retrieve_graph_context]

SYSTEM_PROMPT = """Você é um assistente especializado em análise de documentos com acesso a um grafo de conhecimento.

Você possui DOIS tools:
1. retrieve_graph_context — recupera entidades e relações estruturadas do grafo de conhecimento
2. retrieve_vector_context — recupera trechos de texto por similaridade semântica

Para responder bem:
- Sempre use retrieve_graph_context primeiro para entender as relações entre conceitos
- Use retrieve_vector_context para obter detalhes textuais complementares
- Integre ambas as fontes na sua resposta
- Cite explicitamente quais entidades e relações do grafo embasam sua resposta
- Responda sempre em português, mesmo que a pergunta seja em outro idioma"""


class GraphRAGState(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools = llm.bind_tools(tools)


def agent_node(state: GraphRAGState) -> GraphRAGState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: GraphRAGState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(GraphRAGState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
agent = workflow.compile(checkpointer=checkpointer)


@traceable(name="graph-rag-query", run_type="chain")
def query_graph_rag(
    question: str,
    thread_id: Optional[str] = None,
    callbacks=None,
) -> Dict[str, Any]:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if callbacks:
        config["callbacks"] = callbacks

    events = list(agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="values",
    ))
    final_event = events[-1]
    answer = extract_response_text(final_event["messages"][-1])
    retrieved_docs = vector_store.similarity_search(question, k=3)
    contexts = [doc.page_content for doc in retrieved_docs]
    return {"question": question, "answer": answer, "contexts": contexts}


def run_ragas(ragas_data, llm_eval, embeddings_eval):
    return avaliar_com_ragas(ragas_data, llm_eval, embeddings_eval)


def salvar(df, nome_base="graph-rag"):
    salvar_resultados(df, nome_base)


def main():
    print(f"Carregando documentos de {DOCS_DIR} ...")
    loader = DirectoryLoader(path=DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"{len(docs)} páginas carregadas")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    for i, split in enumerate(all_splits):
        split.metadata["chunk_id"] = f"chunk_{i}"

    if vector_store._collection.count() == 0:
        print(f"Adicionando {len(all_splits)} chunks ao Chroma em batches...")
        batch_size = 500
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            vector_store.add_documents(documents=batch)
            print(f"  {min(i + batch_size, len(all_splits))}/{len(all_splits)} chunks adicionados")

        print("Ingestão concluída!")
    else:
        print(f"Coleção existente com {vector_store._collection.count()} chunks. Pulando ingestão.")

    max_chunks = min(len(all_splits), 20)
    print(f"\nExtraindo entidades de {max_chunks} chunks para o knowledge graph...")
    for i, chunk in enumerate(all_splits[:max_chunks]):
        print(f"  [{i+1}/{max_chunks}] Chunk {chunk.metadata.get('chunk_id')}...", end=" ")
        extraction = extract_entities_from_chunk(chunk)
        ingest_chunk_into_graph(chunk, extraction)
        n_ents = len(extraction.get("entities", []))
        n_rels = len(extraction.get("relations", []))
        print(f"{n_ents} entidades, {n_rels} relações")

    stats = knowledge_graph.stats
    print(f"\nGrafo construído: {stats['nodes']} nós | {stats['edges']} arestas")

    for run in range(5):
        print(f"\n=== RODADA {run + 1}/5 ===")
        eval_llm = build_ragas_llm()

        print("Coletando respostas para avaliação RAGAS...")
        ragas_data = []

        for i, query in enumerate(test_queries):
            print(f"  [{i + 1}/{len(test_queries)}] {query}")
            tracker, started_at = start_usage_tracker()
            result = query_graph_rag(query, callbacks=[tracker])
            ragas_item = {
                "question": result["question"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": ground_truths[i],
            }
            ragas_item.update(finish_usage_tracker(tracker, started_at))
            ragas_data.append(ragas_item)

        df_resultado = run_ragas(ragas_data, eval_llm, embeddings)
        salvar(df_resultado, nome_base=f"graph-rag-run-{run + 1}")


if __name__ == "__main__":
    main()

