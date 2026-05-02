import os
import json
from itertools import count
from typing import List, Dict, Any, Set

import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from datasets import Dataset

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

load_dotenv()

DOCS_DIR = "./docs/"
PERSIST_DIR = "./chroma_graph_db"

RESULTS_BASE_DIR = "results"
N_RUNS = 15

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 500

TOP_K = 5
GRAPH_HOPS = 1

BASE_URL = os.getenv("DO_BASE_URL")
API_KEY = os.getenv("DO_API_KEY")
MODEL = os.getenv("DO_MODEL")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
    "LANGSMITH_ENDPOINT",
    "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT",
    "benchmark-graph-rag"
)


# ============================================================
# SUAS PERGUNTAS E RESPOSTAS ESPERADAS
# ============================================================

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
    "Operadores aritméticos são o conjunto de símbolos que representam as operações básicas da matemática (por exemplo: + para adição, - para subtração, * para multiplicação e / para divisão). Para potenciação e radiciação, o livro indica o uso das palavras‑chave pot e rad.",
    "Operadores relacionais são usados para realizar comparações entre dois valores de mesmo tipo primitivo. Esses valores podem ser constantes, variáveis ou expressões aritméticas, e esses operadores são comuns na construção de equações.",

    # DIFÍCEIS
    "Uma expressão lógica é aquela cujos operadores são lógicos ou relacionais e cujos operandos são relações, variáveis ou constantes do tipo lógico.",
    "Um contador é um modo de contagem feito com a ajuda de uma variável com um valor inicial, que é incrementada a cada repetição. Incrementar significa somar um valor constante (normalmente 1) a cada repetição.",
    "A estrutura de repetição ‘repita ... até’ permite que um bloco (ou ação primitiva) seja repetido até que uma determinada condição seja verdadeira. Pela sintaxe da estrutura, o bloco é executado pelo menos uma vez, independentemente da validade inicial da condição."
]


# ============================================================
# MODELOS
# ============================================================

def criar_llm():
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0
    )


def criar_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# DOCUMENTOS E CHUNKS
# ============================================================

def carregar_pdfs(docs_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def dividir_em_chunks(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")
        chunk.metadata["page"] = chunk.metadata.get("page", None)

    return chunks


# ============================================================
# VECTORSTORE
# ============================================================

def criar_vectorstore(embeddings):
    return Chroma(
        collection_name="graph_rag_collection",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def indexar_documentos_se_necessario(vectordb, chunks: List[Document]):
    if vectordb._collection.count() > 0:
        print(
            f"Coleção existente com {vectordb._collection.count()} chunks. "
            "Pulando ingestão vetorial."
        )
        return

    print(f"Adicionando {len(chunks)} chunks ao Chroma em batches...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectordb.add_documents(batch)

        print(
            f"  {min(i + BATCH_SIZE, len(chunks))}/"
            f"{len(chunks)} chunks adicionados"
        )

    print("Ingestão vetorial concluída.")


# ============================================================
# EXTRAÇÃO DE RELAÇÕES DO GRAFO
# ============================================================

def extrair_relacoes_do_chunk(chunk: Document, llm) -> List[Dict[str, str]]:
    prompt = f"""
Você é um extrator de relações para Graph RAG.

Extraia relações conceituais explícitas do texto abaixo.

Retorne APENAS JSON válido no formato:

[
  {{
    "source": "conceito origem",
    "target": "conceito destino",
    "relation": "tipo da relação",
    "description": "descrição curta da relação"
  }}
]

Regras:
- Não invente relações.
- Extraia apenas relações sustentadas pelo texto.
- Use nomes curtos e consistentes.
- Se não houver relações relevantes, retorne [].

Texto:

{chunk.page_content}
"""

    response = llm.invoke(prompt).content

    try:
        data = json.loads(response)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def normalizar_no(nome: str) -> str:
    return nome.strip().lower()


def construir_grafo(chunks: List[Document], llm) -> nx.DiGraph:
    graph = nx.DiGraph()

    for i, chunk in enumerate(chunks):
        print(f"Extraindo relações do chunk {i + 1}/{len(chunks)}...")

        chunk_id = chunk.metadata["chunk_id"]
        relacoes = extrair_relacoes_do_chunk(chunk, llm)

        for rel in relacoes:
            source_raw = rel.get("source", "").strip()
            target_raw = rel.get("target", "").strip()

            if not source_raw or not target_raw:
                continue

            source = normalizar_no(source_raw)
            target = normalizar_no(target_raw)

            relation = rel.get("relation", "relaciona-se com").strip()
            description = rel.get("description", "").strip()

            if source not in graph:
                graph.add_node(
                    source,
                    name=source_raw,
                    chunk_ids=set()
                )

            if target not in graph:
                graph.add_node(
                    target,
                    name=target_raw,
                    chunk_ids=set()
                )

            graph.nodes[source]["chunk_ids"].add(chunk_id)
            graph.nodes[target]["chunk_ids"].add(chunk_id)

            graph.add_edge(
                source,
                target,
                relation=relation,
                description=description,
                chunk_id=chunk_id
            )

    print(
        f"Grafo construído com {graph.number_of_nodes()} nós "
        f"e {graph.number_of_edges()} relações."
    )

    return graph


# ============================================================
# RECUPERAÇÃO GRAPH RAG
# ============================================================

def recuperar_chunks_vetoriais(query: str, vectordb, top_k: int = TOP_K):
    retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k}
    )
    return retriever.invoke(query)


def encontrar_nos_semente(graph: nx.DiGraph, docs_recuperados: List[Document]) -> Set[str]:
    chunk_ids = {
        doc.metadata.get("chunk_id")
        for doc in docs_recuperados
        if doc.metadata.get("chunk_id") is not None
    }

    nos_semente = set()

    for node, data in graph.nodes(data=True):
        node_chunk_ids = data.get("chunk_ids", set())

        if chunk_ids.intersection(node_chunk_ids):
            nos_semente.add(node)

    return nos_semente


def expandir_nos_do_grafo(
    graph: nx.DiGraph,
    nos_semente: Set[str],
    hops: int = GRAPH_HOPS
) -> Set[str]:
    nos_expandidos = set(nos_semente)

    for _ in range(hops):
        camada_atual = list(nos_expandidos)

        for node in camada_atual:
            if node not in graph:
                continue

            nos_expandidos.update(graph.successors(node))
            nos_expandidos.update(graph.predecessors(node))

    return nos_expandidos


def converter_grafo_em_contexto(
    graph: nx.DiGraph,
    nos: Set[str]
) -> List[str]:
    subgraph = graph.subgraph(nos)
    contextos = []

    for source, target, data in subgraph.edges(data=True):
        source_name = subgraph.nodes[source].get("name", source)
        target_name = subgraph.nodes[target].get("name", target)

        relation = data.get("relation", "relaciona-se com")
        description = data.get("description", "")

        linha = (
            f"{source_name} --[{relation}]--> {target_name}. "
            f"{description}"
        ).strip()

        contextos.append(linha)

    return contextos


def recuperar_contextos_do_grafo(
    graph: nx.DiGraph,
    docs_recuperados: List[Document],
    hops: int = GRAPH_HOPS
) -> List[str]:
    nos_semente = encontrar_nos_semente(graph, docs_recuperados)
    nos_expandidos = expandir_nos_do_grafo(graph, nos_semente, hops)
    return converter_grafo_em_contexto(graph, nos_expandidos)


# ============================================================
# GERAÇÃO DE RESPOSTA
# ============================================================

def gerar_resposta_graph_rag(
    query: str,
    contextos_textuais: List[str],
    contextos_grafo: List[str],
    llm
) -> str:
    contexto_textual = "\n\n".join(contextos_textuais)
    contexto_grafo = "\n".join(contextos_grafo)

    prompt = f"""
Você deve responder usando SOMENTE os contextos fornecidos.

Use tanto o CONTEXTO TEXTUAL quanto o CONTEXTO DE GRAFO.

Se a resposta não estiver nos contextos, diga:
"A informação não está presente no contexto."

CONTEXTO TEXTUAL:
{contexto_textual}

CONTEXTO DE GRAFO:
{contexto_grafo}

PERGUNTA:
{query}

RESPOSTA:
"""

    return llm.invoke(prompt).content


def graph_rag_query(
    query: str,
    vectordb,
    graph: nx.DiGraph,
    llm,
    top_k: int = TOP_K,
    graph_hops: int = GRAPH_HOPS
) -> Dict[str, Any]:
    docs_recuperados = recuperar_chunks_vetoriais(
        query=query,
        vectordb=vectordb,
        top_k=top_k
    )

    contextos_textuais = [
        doc.page_content
        for doc in docs_recuperados
    ]

    contextos_grafo = recuperar_contextos_do_grafo(
        graph=graph,
        docs_recuperados=docs_recuperados,
        hops=graph_hops
    )

    todos_contextos = contextos_textuais + contextos_grafo

    resposta = gerar_resposta_graph_rag(
        query=query,
        contextos_textuais=contextos_textuais,
        contextos_grafo=contextos_grafo,
        llm=llm
    )

    return {
        "question": query,
        "answer": resposta,
        "contexts": todos_contextos,
        "text_contexts": contextos_textuais,
        "graph_contexts": contextos_grafo
    }


# ============================================================
# DATASET PARA RAGAS
# ============================================================

def gerar_dados_ragas(
    test_queries: List[str],
    ground_truths: List[str],
    vectordb,
    graph: nx.DiGraph,
    llm
) -> List[Dict[str, Any]]:
    if len(test_queries) != len(ground_truths):
        raise ValueError(
            "test_queries e ground_truths precisam ter o mesmo tamanho."
        )

    ragas_data = []

    print("Coletando respostas Graph RAG para avaliação RAGAS...")

    for i, query in enumerate(test_queries):
        print(f"  [{i + 1}/{len(test_queries)}] {query}")

        result = graph_rag_query(
            query=query,
            vectordb=vectordb,
            graph=graph,
            llm=llm
        )

        ragas_data.append({
            "question": query,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": ground_truths[i],
            "text_contexts_count": len(result["text_contexts"]),
            "graph_contexts_count": len(result["graph_contexts"])
        })

    return ragas_data


# ============================================================
# AVALIAÇÃO RAGAS
# ============================================================

def run_ragas(ragas_data: List[Dict[str, Any]], llm, embeddings):
    dataset_data = []

    for item in ragas_data:
        dataset_data.append({
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"]
        })

    dataset = Dataset.from_list(dataset_data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas()

    return result, df


# ============================================================
# EXPORTAÇÃO ITERÁVEL
# ============================================================

def salvar(df: pd.DataFrame, nome_base: str = "graph-rag-run") -> str:
    if not hasattr(salvar, "_results_dir"):
        base_dir = "results"

        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=False)
            salvar._results_dir = base_dir
        else:
            for n in count(2):
                candidate = f"{base_dir}_{n}"
                if not os.path.exists(candidate):
                    os.makedirs(candidate, exist_ok=False)
                    salvar._results_dir = candidate
                    break

        print(f"Resultados desta execução serão salvos em: {salvar._results_dir}")

    for i in count(1):
        caminho = os.path.join(
            salvar._results_dir,
            f"{nome_base}_{i}.csv"
        )

        if not os.path.exists(caminho):
            df.to_csv(
                caminho,
                index=False,
                encoding="utf-8-sig",
                sep=";"
            )
            print(f"Salvo em: {caminho}")
            return caminho


# ============================================================
# PIPELINE COMPLETO
# ============================================================

def preparar_graph_rag():
    llm = criar_llm()
    embeddings = criar_embeddings()

    print("Carregando documentos...")
    docs = carregar_pdfs(DOCS_DIR)

    print("Dividindo documentos em chunks...")
    chunks = dividir_em_chunks(docs)

    print("Criando vectorstore...")
    vectordb = criar_vectorstore(embeddings)

    print("Indexando documentos, se necessário...")
    indexar_documentos_se_necessario(vectordb, chunks)

    print("Construindo grafo...")
    graph = construir_grafo(chunks, llm)

    return {
        "llm": llm,
        "embeddings": embeddings,
        "vectordb": vectordb,
        "graph": graph,
        "chunks": chunks
    }


def executar_runs_graph_rag(
    test_queries: List[str],
    ground_truths: List[str],
    pipeline: Dict[str, Any],
    n_runs: int = 15
):
    dfs = []

    for run in range(1, n_runs + 1):
        print(f"\n=== RODADA {run}/{n_runs} ===")

        ragas_data = gerar_dados_ragas(
            test_queries=test_queries,
            ground_truths=ground_truths,
            vectordb=pipeline["vectordb"],
            graph=pipeline["graph"],
            llm=pipeline["llm"]
        )

        _, df = run_ragas(
            ragas_data=ragas_data,
            llm=pipeline["llm"],
            embeddings=pipeline["embeddings"]
        )

        df["run"] = run

        salvar(
            df=df,
            nome_base=f"graph-rag-run-{run}"
        )

        dfs.append(df)

    df_summary = pd.concat(dfs, ignore_index=True)

    salvar(
        df=df_summary,
        nome_base="graph-rag-all-runs"
    )

    return df_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pipeline = preparar_graph_rag()

    df_resultados = executar_runs_graph_rag(
        test_queries=test_queries,
        ground_truths=ground_truths,
        pipeline=pipeline,
        n_runs=15
    )