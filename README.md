# DuMF-Agent: Dual-Channel Memory Framework for Long-Term Conversational Agents

A long-term memory architecture for conversational AI that addresses memory fragmentation, temporal confusion, and cross-session reasoning instability through unified memory representation, retrieval-reading closed-loop, and temporal version consistency mechanisms.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DuMF-Agent Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   User      │    │              Dual-Channel Memory                 │   │
│  │   Query     │───▶│  ┌────────────────┐  ┌────────────────────────┐ │   │
│  └─────────────┘    │  │  RAW Channel   │  │  CONSOLIDATED Channel  │ │   │
│                     │  │  (Evidence)    │  │  (SimpleFact + Triple) │ │   │
│                     │  └────────────────┘  └────────────────────────┘ │   │
│                     └──────────────────────────────────────────────────┘   │
│                                      │                                      │
│                     ┌────────────────▼────────────────┐                    │
│                     │      Hybrid Retrieval           │                    │
│                     │  • Query Expansion              │                    │
│                     │  • Vector + BM25 + Multi-hop    │                    │
│                     │  • Unified Re-ranking           │                    │
│                     └────────────────┬────────────────┘                    │
│                                      │                                      │
│                     ┌────────────────▼────────────────┐                    │
│                     │      Context Construction       │                    │
│                     │  • Version Detection            │                    │
│                     │  • Temporal Filtering           │                    │
│                     │  • Evidence Organization        │                    │
│                     └────────────────┬────────────────┘                    │
│                                      │                                      │
│                     ┌────────────────▼────────────────┐                    │
│                     │         LLM Generation          │                    │
│                     └─────────────────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Three-Layer Memory Representation**: Evidence (RAW/TextUnit), Language (SimpleFact), and Structure (Structured Triple) layers with cross-references for traceable reasoning
- **Retrieval-Reading Closed-Loop**: Query expansion, hybrid retrieval (vector + BM25 + multi-hop), and unified re-ranking with confidence-aware scoring
- **Temporal Version Consistency**: Append-only storage with dynamic version detection to distinguish current vs. historical facts

## Installation

### Prerequisites

- Python 3.9+
- Neo4j 5.x (local or Aura cloud)
- CUDA-compatible GPU (optional, for local embeddings)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DuMF-Agent.git
cd DuMF-Agent
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

5. Initialize Neo4j schema:
```bash
python utils/init_neo4j_schema.py
python utils/create_fulltext_index.py
```

6. (Optional) Start local embedding server:
```bash
python embedding_server.py
```

## Data Preparation

This project uses the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark for evaluation.

1. Download the dataset from the official repository
2. Place the files in `data/long_memory_eval/`:
   - `sampled_test_questions.json` - Sample setting (~500 instances, 2-4k tokens each)
   - `medium_test_questions.json` - Hard setting (~500 instances, ~115k tokens each)

Directory structure:
```
data/
├── long_memory_eval/
│   ├── sampled_test_questions.json
│   └── medium_test_questions.json
├── books/           # Character profiles (optional)
└── world_knowledge/ # World knowledge base (optional)
```

## Configuration

Key parameters in `.env`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `GRAPHRAG_CHAT_MODEL` | LLM for generation | gpt-4o-mini |
| `GRAPHRAG_EMBEDDING_MODEL` | Embedding model | BAAI/bge-m3 |
| `NEO4J_URI` | Neo4j connection URI | neo4j://127.0.0.1:7687 |
| `EVIDENCE_FILTER_LEVEL` | Evidence filtering strictness | lenient |

See `.env.example` for full configuration options.

## Usage

### Basic Usage

```python
from agent.agent import DuMFAgent

# Initialize agent
agent = DuMFAgent(agent_id="user_001")

# Process conversation
response = agent.chat("What did we discuss about the project last week?")
```

### Running Evaluation

```bash
# Run on LongMemEval sample setting
python test/Long_Memory_test.py

# Results will be saved to result/simple/long_memory_results.jsonl
```

### Embedding Server

For local embedding (recommended for development):
```bash
# Start the embedding server first
python embedding_server.py

# Configure in .env:
# GRAPHRAG_EMBEDDING_API_BASE=http://127.0.0.1:8000
```

For online embedding API, configure SiliconFlow or other providers in `.env`.

## Project Structure

```
DuMF-Agent/
├── agent/                  # Core agent implementation
│   ├── agent.py           # Main agent class
│   ├── simple_retriever.py # Hybrid retrieval system
│   └── context_builder.py  # Context construction
├── memory/                 # Memory system
│   ├── dual_memory_system.py
│   ├── structured_memory.py
│   └── stores.py
├── temporal_reasoning/     # Temporal reasoning module
│   ├── executor.py
│   └── intent_router.py
├── prompts/               # Prompt templates
├── utils/                 # Utility functions
└── test/                  # Test scripts
```

## Evaluation Results

Performance on LongMemEval benchmark:

| Method | Overall Acc. (Sample) | Overall Acc. (Hard) |
|--------|----------------------|---------------------|
| LLM    | 0.7500               | 0.4138              |
| RAG    | 0.6724               | 0.5172              |
| Mem0   | 0.5690               | 0.2759              |
| GA     | 0.5862               | 0.2414              |
| **DuMF-Agent** | **0.7241**   | **0.5517**          |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LongMemEval benchmark for evaluation framework
- Neo4j for graph database support
