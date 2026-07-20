# Graph RAG

Sistema de perguntas e respostas sobre documentos PDF (apostilas de lógica de programação e algoritmos) que combina busca vetorial com um **Knowledge Graph extraído automaticamente por LLM**, consultado por um agente orquestrado via LangGraph, com avaliação automática de qualidade via RAGAS.

## Contextualização

RAG (Retrieval-Augmented Generation) é o padrão de recuperar trechos relevantes de uma base documental e injetá-los no prompt de um LLM para fundamentar a resposta. "Graph RAG" designa a família de técnicas que complementam esse retrieval textual com um grafo de conhecimento — entidades e relações entre elas — como fonte adicional de contexto estruturado.

Este projeto **não** é o GraphRAG oficial da Microsoft: não há sumarização de comunidades, nem algoritmos de clustering (Leiden/Louvain), nem busca "local"/"global" no sentido daquela técnica. É uma arquitetura própria e mais simples: um Knowledge Graph é construído **automaticamente**, via extração de entidades e relações por LLM sobre uma amostra dos chunks indexados, armazenado em memória com `networkx`, e consultado por um **agente LangGraph** que decide dinamicamente quando usar o grafo e quando usar busca vetorial tradicional, através de duas tools.

## Arquitetura do pipeline

```
docs/*.pdf
    │  PyPDFLoader + DirectoryLoader
    ▼
Documentos (1 por página)
    │  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200, add_start_index=True)
    ▼
Chunks (com chunk_id sequencial)
    │
    ├──────────────────────────────┐
    ▼                               ▼
OpenAIEmbeddings                Primeiros 20 chunks
    │                               │  EXTRACTION_SYSTEM + EXTRACTION_TEMPLATE (LLM → JSON)
    ▼                               ▼
Chroma (persistente,          Entidades + relações extraídas
batches de 500)                    │
    │                               ▼
    │                        KnowledgeGraph (networkx.DiGraph, em memória)
    │                               │
    ▼                               ▼
retrieve_vector_context (k=3)   retrieve_graph_context (BFS depth=2, até 40 nós)
    │                               │
    └───────────────┬───────────────┘
                     ▼
       Agente LangGraph (StateGraph: agent ⇄ tools)
       SYSTEM_PROMPT: usar o grafo primeiro, depois o vetor
                     │
                     ▼
                 Resposta
                     │
                     ▼
         Avaliação RAGAS (5 rodadas)
                     │
                     ▼
        results/graph-rag-run-N_i.csv
```

| Etapa | Função / arquivo |
|---|---|
| Ingestão + chunking | bloco inicial de `main()` — `main.py:353-363` |
| Embeddings | `build_embeddings()` — `rag_settings.py:67-71` |
| Indexação vetorial | `main()` — `main.py:365-373` |
| Extração de entidades/relações | `extract_entities_from_chunk()` — `main.py:201-213` |
| Construção do grafo | `ingest_chunk_into_graph()` — `main.py:216-243`, classe `KnowledgeGraph` — `main.py:109-177` |
| Tool de busca vetorial | `retrieve_vector_context()` — `main.py:246-254` |
| Tool de busca no grafo | `retrieve_graph_context()` — `main.py:257-269` |
| Orquestração do agente | `StateGraph` (`agent_node`, `should_continue`, `ToolNode`) — `main.py:288-316` |
| Execução de uma pergunta | `query_graph_rag()` — `main.py:319-341` |
| Avaliação | `run_ragas()` — `rag_settings.py:277-298` |
| Persistência dos resultados | `salvar()` — `rag_settings.py:301-336` |
| Orquestração / loop principal | `main()` — `main.py:352-415` |

## Detalhes técnicos

### Prompts

**Extração de entidades/relações** (`main.py:182-198`), aplicado sobre os primeiros 1500 caracteres de cada chunk (`chunk.page_content[:1500]`):

```
EXTRACTION_SYSTEM = """Você é um extrator especializado de entidades e relações.
Retorne APENAS um JSON válido, sem texto adicional, sem markdown."""

EXTRACTION_TEMPLATE = """Analise o texto e extraia entidades e relações.

Retorne SOMENTE este JSON (sem blocos de código, sem explicações):
{
  "entities": [
    {"id": "e1", "name": "Nome da Entidade", "type": "Concept|Person|Organization|Location|Event", "description": "descrição breve"}
  ],
  "relations": [
    {"source": "e1", "target": "e2", "type": "RELATES_TO|IS_A|PART_OF|CAUSES|DEFINES", "description": "como se relacionam"}
  ]
}

Texto:
{text}"""
```

A resposta do LLM é limpa de eventuais cercas de código via regex e parseada como JSON; em caso de falha, o resultado é tratado como `{"entities": [], "relations": []}` (`main.py:201-213`).

**Prompt de sistema do agente** (`main.py:274-285`):

```
Você é um assistente especializado em análise de documentos com acesso a um grafo de conhecimento.

Você possui DOIS tools:
1. retrieve_graph_context — recupera entidades e relações estruturadas do grafo de conhecimento
2. retrieve_vector_context — recupera trechos de texto por similaridade semântica

Para responder bem:
- Sempre use retrieve_graph_context primeiro para entender as relações entre conceitos
- Use retrieve_vector_context para obter detalhes textuais complementares
- Integre ambas as fontes na sua resposta
- Cite explicitamente quais entidades e relações do grafo embasam sua resposta
- Responda sempre em português, mesmo que a pergunta seja em outro idioma
```

Não há prompt de sumarização de comunidades — o projeto não implementa detecção de comunidades/clustering.

### Chunking

- Biblioteca: `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- `chunk_size=1000`, `chunk_overlap=200`, `add_start_index=True` (`main.py:358-360`).
- Cada chunk recebe `metadata["chunk_id"] = f"chunk_{i}"` sequencial.
- Carregamento via `DirectoryLoader(path=DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)`.
- **Apenas os primeiros `min(len(chunks), 20)` chunks** são usados para extração de entidades/relações (limite fixo no código, por custo) — os demais entram só no índice vetorial. Ou seja, o grafo cobre uma fração pequena do corpus, enquanto o índice vetorial cobre tudo.

### Embeddings

```python
def build_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=get_openai_api_key(),
    )
```

Modelo padrão `text-embedding-3-large` (OpenAI), dimensão nativa 3072 não fixada explicitamente no código. Usado apenas para o índice vetorial — o grafo não usa embeddings.

### Banco vetorial

- **ChromaDB**, persistido em disco: `persist_directory=CHROMA_PERSIST_DIR` (default `./chroma_graph_db_openai`), `collection_name=CHROMA_COLLECTION_NAME` (default `graph_collection_openai`).
- Ingestão em batches de 500, condicional a `vector_store._collection.count() == 0`.

### Configuração de grafo

**Banco de grafos**: `networkx.DiGraph`, **em memória**, sem persistência em disco entre execuções — não é Neo4j nem qualquer banco de grafos externo.

**Schema de nós** (`EntityNode`, `main.py:91-97`): `id`, `name`, `type` (um de `Concept`, `Person`, `Organization`, `Location`, `Event`, conforme o prompt de extração), `description`, `chunk_ids` (lista de chunks de origem daquela entidade).

**Schema de arestas** (`RelationEdge`, `main.py:100-106`): `source_id`, `target_id`, `relation_type` (um de `RELATES_TO`, `IS_A`, `PART_OF`, `CAUSES`, `DEFINES`), `description`, `weight` (default `1.0`, não usado no cálculo de relevância atualmente).

**ID global de entidade**: gerado deterministicamente como `f"{nome.lower().replace(' ', '_')}_{tipo.lower()}"` (`main.py:221`) — isso deduplica entidades com o mesmo nome+tipo encontradas em chunks diferentes; se a entidade já existe, o novo `chunk_id` é apenas anexado a ela.

**Construção**: para cada um dos 20 primeiros chunks, `extract_entities_from_chunk()` chama o LLM extrator e `ingest_chunk_into_graph()` adiciona as entidades/relações retornadas ao grafo (`main.py:377-385`).

**Consulta/traversal**:
- `get_neighbors(entity_id, depth=2)` (`main.py:144-155`) — BFS bidirecional (sucessores + predecessores) até profundidade 2.
- `build_context(node_ids)` (`main.py:157-173`) — serializa o subgrafo (entidades com tipo/descrição + relações com origem/tipo/destino/descrição) em texto markdown para o LLM.
- **Entity linking na busca** (`retrieve_graph_context`, `main.py:257-269`): busca ingênua por palavras da query com mais de 3 caracteres, contra os nomes das entidades (correspondência exata ou parcial via substring, case-insensitive, em `find_entity_by_name`), sem NER real. Os vizinhos de cada entidade encontrada são agregados, deduplicados e limitados a 40 nós.
- **Não há**: algoritmos de detecção de comunidade (Leiden/Louvain), PageRank, centralidade, nem qualquer "resumo global" do estilo GraphRAG clássico.

### Parâmetros de recuperação

- Busca vetorial (`retrieve_vector_context` e a busca extra usada para o RAGAS): `similarity_search(query, k=3)`.
- Busca no grafo: profundidade de traversal fixa `depth=2`, limite de `40` nós após deduplicação; sem score de relevância, sem re-ranking.
- Decisão de qual tool usar e quando: delegada ao **agente** (tool-calling do LLM), não é uma regra hardcoded — o `SYSTEM_PROMPT` apenas instrui a ordem preferida (grafo primeiro, depois vetor), mas o modelo pode não seguir essa instrução à risca.
- `MemorySaver` como checkpointer do LangGraph, associado a um `thread_id` por chamada de `query_graph_rag()` (gerado novo a cada pergunta no benchmark, salvo se um `thread_id` for passado explicitamente).
- Os `contexts` usados na avaliação RAGAS vêm de uma **busca vetorial adicional e independente** (`main.py:339`, `k=3`), separada do que as tools do agente efetivamente recuperaram durante a conversa — os dois conjuntos de contexto podem divergir.

### Versões das bibliotecas

`requirements.txt` não fixa versões exatas (só um mínimo):

| Biblioteca | Versão |
|---|---|
| langchain | não pinada |
| langchain-community | não pinada |
| langchain-openai | `>=1.1.11` |
| langchain-text-splitters | não pinada |
| langgraph | não pinada |
| openai | não pinada |
| chromadb | não pinada |
| networkx | não pinada |
| pypdf | não pinada |
| datasets | não pinada |
| python-dotenv | não pinada |
| ragas | não pinada |
| langsmith | não pinada |

Não há `pyproject.toml` nem lockfile. Não há `neo4j` nem o pacote `graphrag` da Microsoft entre as dependências — confirma que este não é o GraphRAG oficial.

## Requisitos

- Python 3.10+
- Conta OpenAI com acesso à API
- Conta LangSmith, para rastreamento (tracing) do fluxo e cálculo de uso de tokens

## Replicabilidade / Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Configuração

Crie um `.env` a partir de `.env.example`:

```env
OPENAI_API_KEY=sk-sua_chave_openai
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_REASONING_EFFORT=medium

DOCS_DIR=./docs/

CHROMA_PERSIST_DIR=./chroma_graph_db_openai
CHROMA_COLLECTION_NAME=graph_collection_openai

LANGCHAIN_TRACING_V2=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=benchmark-graph-rag
```

| Variável | Default | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — (obrigatória) | Chave da API OpenAI. |
| `OPENAI_MODEL` | `gpt-5.5` | Modelo usado pelo agente, pela extração de entidades e pela avaliação RAGAS. Reportado como está no código/`.env.example`. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Modelo de embeddings. |
| `OPENAI_REASONING_EFFORT` | `medium` | Parâmetro `reasoning_effort` do `ChatOpenAI` (Responses API). |
| `DOCS_DIR` | `./docs/` | Pasta com os PDFs a indexar. |
| `CHROMA_PERSIST_DIR` | `./chroma_graph_db_openai` | Diretório de persistência do índice vetorial. |
| `CHROMA_COLLECTION_NAME` | `graph_collection_openai` | Nome da coleção no Chroma. |
| `LANGCHAIN_TRACING_V2` | `false` | Ativa tracing no LangSmith. |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | Endpoint do LangSmith. |
| `LANGCHAIN_API_KEY` | — | Chave do LangSmith. |
| `LANGCHAIN_PROJECT` | `benchmark-graph-rag` | Nome do projeto no LangSmith. |

## Uso

Coloque os PDFs em `docs/` (já populada com 7 apostilas/livros sobre algoritmos e lógica de programação) e execute:

```bash
python main.py
```

O script:
1. Carrega e faz o chunking dos PDFs de `DOCS_DIR`.
2. Indexa todos os chunks no Chroma (pula se a coleção já existir).
3. Extrai entidades/relações dos primeiros 20 chunks e constrói o Knowledge Graph em memória, imprimindo o total de nós/arestas ao final.
4. Roda **5 rodadas** das mesmas **10 perguntas de benchmark** fixas no código (`test_queries`/`ground_truths` em `main.py`) — como o grafo é reconstruído apenas uma vez no início e o índice vetorial é idempotente, as rodadas repetem apenas a etapa de pergunta/resposta/avaliação.
5. Para cada pergunta, o agente decide quais tools usar (grafo e/ou vetor) e gera a resposta.
6. Avalia cada rodada com RAGAS e salva um CSV por rodada em `results/` (ou `results_2/`, `results_3/`... se a pasta já existir).

## Estrutura do projeto

```
graph-rag/
├── .env.example
├── README.md
├── requirements.txt
├── main.py              # pipeline Graph RAG (extração de KG + agente LangGraph) + benchmark RAGAS
├── rag_settings.py       # utilitários compartilhados: env, LLM/embeddings, tracking de uso, RAGAS, salvar CSV
├── graph_rag.ipynb        # variante exploratória (ver Notas)
└── docs/                 # 7 PDFs (livros de algoritmos/lógica de programação em PT-BR) usados como corpus
```

Gerados em runtime (fora do controle de versão): `chroma_graph_db_openai/` (índice vetorial) e `results*/` (CSVs). O Knowledge Graph em si não é persistido — é reconstruído do zero a cada execução do script.

## Avaliação e resultados

Métricas RAGAS calculadas a cada rodada: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Cada linha do CSV também traz `answer_response_time_seconds`, `answer_input_tokens`, `answer_output_tokens` e `answer_total_tokens`, medidos por pergunta via `TokenUsageTracker`.

## Notas e limitações

- A extração de entidades/relações é limitada aos primeiros 20 chunks do corpus (por custo) — o grafo não representa a totalidade dos documentos indexados, apenas essa amostra inicial.
- O entity linking usado na busca do grafo é um matching textual simples (substring, case-insensitive), não NER ou embeddings — nomes de entidades muito curtos (≤3 caracteres) ou com grafia diferente do texto original podem não ser encontrados.
- O Knowledge Graph é reconstruído em memória a cada execução do script; não há persistência em disco nem cache entre rodadas.
- Os `contexts` usados na avaliação RAGAS vêm de uma busca vetorial independente das tools do agente — pode haver divergência entre o que o agente efetivamente usou para responder e o que é avaliado como contexto.
- `graph_rag.ipynb` é uma variante exploratória com stack diferente: embeddings `HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)`, `InMemoryVectorStore` (não persistente) em vez de Chroma, e um LLM via endpoint compatível com OpenAI hospedado na DigitalOcean, em vez da stack OpenAI usada em `main.py`. Inclui também visualizações extras (grafo via `networkx`/`matplotlib`, gráficos de métricas RAGAS) não presentes no pipeline oficial.
- Dependências em `requirements.txt` não são pinadas (exceto o mínimo de `langchain-openai`).
