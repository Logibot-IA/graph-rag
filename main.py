import os
import uuid
import json
import re
from itertools import count
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from dataclasses import dataclass, field

import dotenv
import networkx as nx

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain.chat_models import init_chat_model

from transformers import logging as hf_logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

hf_logging.set_verbosity_error()

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "benchmark-graph-rag")

test_queries = [
    "Como o livro Algoritmos: Teoria e Prática, de Cormen, define a notação Θ (Theta) e qual teorema relaciona Θ com as notações O e Ω?",
    "Como Manzano e Oliveira, no livro Algoritmos: Lógica para Desenvolvimento de Programação de Computadores, descrevem o papel do programador de computador e o que é o diagrama de blocos?",
    "Segundo Dilermando Junior e Nakamiti em Algoritmos e Programação de Computadores, qual é a origem do termo \"algoritmo\" e em que consiste o Algoritmo Euclidiano para o cálculo do mdc?",
    "Por que, segundo Sebesta no livro Conceitos de Linguagens de Programação, é importante estudar os conceitos de linguagens de programação mesmo para quem não vai criar uma nova linguagem?",
    "Como Bhargava, no livro Entendendo Algoritmos, define a notação Big O e o que ela estabelece sobre o tempo de execução de um algoritmo?",
    "Segundo Szwarcfiter em Estruturas de Dados e Seus Algoritmos, quais são as complexidades das operações de seleção, inserção, remoção, alteração e construção em um heap?",
    "Como Ascencio, no livro Fundamentos da Programação de Computadores, descreve a plataforma Java, os arquivos gerados na compilação e o papel da Máquina Virtual Java?",
    "Segundo o livro Introdução a Algoritmos e Programação, quais são as três partes que compõem um algoritmo executado em um computador e quais sistemas de representação numérica são utilizados internamente?",
    "Quais são as quatro perguntas que Nilo Menezes, em Introdução à Programação com Python, recomenda que o iniciante responda antes de começar a aprender a programar e qual é, segundo o autor, a maneira mais difícil de aprender?",
    "Quais são os operadores aritméticos não convencionais apresentados por Forbellone em Lógica de Programação e como o autor define o conceito de contador?"
]

ground_truths = [
    "Cormen define que, para uma função g(n), Θ(g(n)) representa o conjunto de funções com limites assintóticos justos: existe um limite superior e inferior do mesmo crescimento. O Teorema 3.1 do livro estabelece que, para quaisquer duas funções f(n) e g(n), tem-se f(n) = Θ(g(n)) se e somente se f(n) = O(g(n)) e f(n) = Ω(g(n)). Em outras palavras, uma função tem ordem Θ exatamente quando possui simultaneamente o mesmo limite assintótico superior (O) e inferior (Ω).",
    "Manzano e Oliveira comparam o programador a um construtor (ou pedreiro especializado), responsável por construir o programa empilhando instruções de uma linguagem como se fossem tijolos, inclusive elaborando a interface gráfica. Além de interpretar o fluxograma desenhado pelo analista, o programador deve detalhar a lógica do programa em nível micro, desenhando uma planta operacional chamada diagrama de blocos (ou diagrama de quadros), seguindo a norma ISO 5807:1985. Essa atividade exige alto grau de atenção e cuidado, pois o descuido pode \"matar\" uma empresa.",
    "Segundo Dilermando Junior e Nakamiti, o termo \"algoritmo\" deriva do nome do matemático persa al-Khwarizmi, considerado por muitos o \"Pai da Álgebra\". No século XII, Adelardo de Bath traduziu uma de suas obras para o latim, registrando o termo como \"Algorithmi\"; originalmente referia-se às regras de aritmética com algarismos indo-arábicos e, posteriormente, passou a designar qualquer procedimento definido para resolver problemas. O Algoritmo Euclidiano, criado por Euclides, calcula o máximo divisor comum (mdc): divide-se a por b, obtendo o resto r; substitui-se a por b e b por r; e repete-se a divisão até que não seja mais possível dividir, sendo o último valor de a o mdc.",
    "Sebesta argumenta que estudar conceitos de linguagens valoriza recursos e construções importantes e estimula o programador a usá-los mesmo quando a linguagem em uso não os suporta diretamente — por exemplo, simulando matrizes associativas de Perl em outra linguagem. Também fornece embasamento para escolher a linguagem mais adequada a cada projeto, evitando que o profissional se restrinja àquela com a qual está mais familiarizado. Por fim, conhecer uma gama mais ampla de linguagens torna o aprendizado de novas linguagens mais fácil, ampliando a capacidade de avaliar trade-offs de projeto.",
    "Bhargava define a notação Big O como uma forma de medir o tempo de execução de um algoritmo no pior caso (pior hipótese), descrevendo o quão rapidamente esse tempo cresce em relação ao tamanho n da entrada. Por exemplo, a pesquisa simples tem tempo O(n) — no pior caso verifica todos os elementos da lista — enquanto a pesquisa binária tem tempo O(log n). Algoritmos com tempos diferentes crescem a taxas muito distintas, e o Big O permite compará-los independentemente do hardware utilizado.",
    "Segundo Szwarcfiter, em um heap o elemento de maior prioridade é sempre a raiz da árvore, e as operações têm os seguintes parâmetros de eficiência: seleção em O(1), pois basta retornar a raiz; inserção em O(log n); remoção em O(log n); alteração em O(log n); e construção em O(n), tempo este inferior ao de uma ordenação. Esses tempos tornam o heap especialmente adequado para implementar listas de prioridades.",
    "Ascencio explica que a tecnologia Java é composta pela linguagem de programação Java e pela plataforma de desenvolvimento Java, com características de simplicidade, orientação a objetos, portabilidade, alta performance e segurança. Os programas são escritos em arquivos de texto com extensão .java e, ao serem compilados pelo compilador javac, geram arquivos .class compostos por bytecodes — código interpretado pela Máquina Virtual Java (JVM). A plataforma Java é composta apenas por software, pois é a JVM que faz a interface entre os programas e o sistema operacional.",
    "O livro descreve que um algoritmo, quando programado em um computador, é constituído por pelo menos três partes: entrada de dados, processamento de dados e saída de dados. Internamente, os computadores digitais utilizam o sistema binário (base 2), com apenas dois algarismos (0 e 1), aproveitando a noção de ligado/desligado ou verdadeiro/falso. Como representações auxiliares, são também utilizados o sistema decimal (base 10), o sistema hexadecimal (base 16, com dígitos 0–9 e A–F) e o sistema octal (base 8).",
    "Menezes propõe que o iniciante responda a quatro perguntas antes de começar: (1) Você quer aprender a programar?; (2) Como está seu nível de paciência?; (3) Quanto tempo você pretende estudar?; (4) Qual o seu objetivo ao programar? Para o autor, a maneira mais difícil de aprender a programar é não querer programar — a vontade deve vir do próprio aluno e não de um professor ou amigo. Programar é uma arte que exige tempo, dedicação e paciência para que a mente se acostume com a nova forma de pensar.",
    "Forbellone apresenta operadores aritméticos não convencionais úteis na construção de algoritmos: pot(x,y) para potenciação (x elevado a y), rad(x) para radiciação (raiz quadrada de x), mod para o resto da divisão (ex.: 9 mod 4 = 1) e div para o quociente da divisão inteira (ex.: 9 div 4 = 2). Um contador é uma variável usada para registrar quantas vezes um trecho de algoritmo é executado: é declarada com um valor inicial e incrementada (somada de uma constante, normalmente 1) a cada repetição, comportando-se como o ponteiro dos segundos de um relógio."
]

llm = init_chat_model(
    model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
    model_provider="openai",
    base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1"),
)

llm_extractor = init_chat_model(
    model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
    model_provider="openai",
    base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1"),
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)


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
    raw = response.content.strip()
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


def query_graph_rag(question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    events = list(agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ))
    final_event = events[-1]
    answer = final_event["messages"][-1].content
    retrieved_docs = vector_store.similarity_search(question, k=3)
    contexts = [doc.page_content for doc in retrieved_docs]
    return {"question": question, "answer": answer, "contexts": contexts}


def run_ragas(ragas_data, llm_eval, embeddings_eval):
    dataset = Dataset.from_list(ragas_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_eval,
        embeddings=embeddings_eval,
    )
    print("\n=== RESULTADOS RAGAS ===")
    print(result)
    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df.to_string())
    return result


def salvar(df, nome_base="graph-rag"):
    os.makedirs("results", exist_ok=True)
    for i in count(1):
        nome = os.path.join("results", f"{nome_base}_{i}.csv")
        if not os.path.exists(nome):
            df.to_csv(nome, index=False, encoding="utf-8-sig", sep=";")
            print(f"Salvo em: {nome}")
            break


def main():
    print("Carregando documentos de ../docs/ ...")
    loader = DirectoryLoader(path="../docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"{len(docs)} páginas carregadas")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    for i, split in enumerate(all_splits):
        split.metadata["chunk_id"] = f"chunk_{i}"

    print(f"Adicionando {len(all_splits)} chunks ao vector store em batches...")
    batch_size = 500
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"  {min(i + batch_size, len(all_splits))}/{len(all_splits)} chunks adicionados")

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

    print("\nColetando respostas para avaliação RAGAS...")
    ragas_data = []
    for i, query in enumerate(test_queries):
        print(f"  [{i+1}/{len(test_queries)}] {query}")
        result = query_graph_rag(query)
        ragas_data.append({
            "question": result["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": ground_truths[i],
        })

    eval_llm = init_chat_model(
        model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
        model_provider="openai",
        base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1"),
    )

    result = run_ragas(ragas_data, eval_llm, embeddings)
    salvar(result.to_pandas())


if __name__ == "__main__":
    main()
