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

### Download Dataset

```bash
# Clone LongMemEval repository
git clone https://github.com/xiaowu0162/LongMemEval.git

# Copy test files to your project
mkdir -p data/long_memory_eval
cp LongMemEval/data/*.json data/long_memory_eval/
```

### Verify Directory Structure

```
data/
└── long_memory_eval/
    ├── longmemeval_oracle.json   # Sample setting
    └── longmemeval_s.json     # Hard setting
```

## Configuration

### Environment Variables (.env)

Copy `.env.example` to `.env` and configure the following:

#### Required Settings

```bash
# LLM API (OpenAI-compatible)
GRAPHRAG_API_BASE=https://api.openai.com/v1
GRAPHRAG_CHAT_API_KEY=sk-your-api-key-here
GRAPHRAG_CHAT_MODEL=gpt-4o-mini

# Cheap LLM for extraction tasks
CHEAP_GRAPHRAG_API_BASE=https://api.openai.com/v1
CHEAP_GRAPHRAG_CHAT_API_KEY=sk-your-api-key-here
CHEAP_GRAPHRAG_CHAT_MODEL=gpt-4o-mini

# Embedding Model
GRAPHRAG_EMBEDDING_API_BASE=http://127.0.0.1:8000  # Local server
GRAPHRAG_EMBEDDING_API_KEY=local
GRAPHRAG_EMBEDDING_MODEL=BAAI/bge-m3

# Neo4j Database
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
```

#### Optional Settings

```bash
# Evidence filtering: strict | medium | lenient
EVIDENCE_FILTER_LEVEL=lenient

# TextUnit fallback: off | order | always
EVIDENCE_TEXTUNIT_FALLBACK_SCOPE=order

# Confidence scores
RAW_REL_CONFIDENCE=0.95
CONSOLIDATED_REL_CONFIDENCE=0.85
CONSOLIDATED_ASSERTS_CONFIDENCE=0.6
```

### Key Parameters in config.py

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SimpleFact k` | 100 | Top-k for SimpleFact retrieval |
| `TextUnit k` | 10 | Top-k for TextUnit retrieval |
| `Fulltext k` | 20 | Top-k for BM25 fulltext search |
| `Multi-hop limit` | 20 | Max nodes in graph expansion |
| `Multi-hop decay` | 0.85 | Score decay per hop |
| `Similarity weight` | 0.7 | Weight for semantic similarity |
| `Confidence weight` | 0.2 | Weight for fact confidence |
| `Channel weight` | 0.1 | Weight for channel priority |
| `Version threshold` | 0.75 | Threshold for version detection |

See `config.py` for all configurable parameters.

## Usage

### Basic Usage

```python
from agent.agent import DuMFAgent

# Initialize agent
agent = DuMFAgent(agent_id="user_001")

# Process conversation
response = agent.chat("What did we discuss about the project last week?")
```

## Running LongMemEval Evaluation

### Test Script: `test/Long_Memory_test.py`

This is the main evaluation script for reproducing LongMemEval benchmark results.

### Quick Start

```bash
# 1. Ensure Neo4j is running
neo4j start

# 2. Initialize database schema
python utils/init_neo4j_schema.py
python utils/create_fulltext_index.py

# 3. (Optional) Start local embedding server
python embedding_server.py  # Run in separate terminal

# 4. Run evaluation
python test/Long_Memory_test.py
```

### Command Line Options

```bash
python test/Long_Memory_test.py [OPTIONS]

Basic Options:
  --data PATH            Path to LongMemEval test file
                         Default: data/long_memory_eval/sampled_test_questions.json
  
  --output PATH          Path to save results (JSONL format)
                         Default: test/long_memory_results.json
  
  --no-debug            Disable debug mode (debug is ON by default)

Slicing Options (for testing subsets):
  --limit N             Process only first N test cases
  
  --start N             Start from index N (0-based)
  
  --end N               End at index N (exclusive)
  
  --indices "0,3,4"     Process specific indices (comma-separated)

Parallel Execution Options (advanced):
  --full                Run full dataset with parallel execution
  
  --parallel            Run local + Aura in parallel with auto split
  
  --queue-init          Initialize a dynamic queue for workers
  
  --queue-worker        Run as a dynamic queue worker
  
  --queue-parallel      Init queue + run workers + merge results
  
  --queue-path PATH     Path to queue file (default: test/long_memory_queue.json)
  
  --queue-results PATH  Path to queue results (default: test/long_memory_results.queue.jsonl)
```

### Example Commands

```bash
# 1. Run on sample setting (default)
python test/Long_Memory_test.py

# 2. Run on hard setting
python test/Long_Memory_test.py \
    --data data/long_memory_eval/medium_test_questions.json \
    --output result/medium/long_memory_results.jsonl

# 3. Test with first 10 cases only
python test/Long_Memory_test.py --limit 10

# 4. Test specific range (cases 0-50)
python test/Long_Memory_test.py --start 0 --end 50

# 5. Test specific cases
python test/Long_Memory_test.py --indices "0,5,10,15"

# 6. Run full dataset with parallel execution
python test/Long_Memory_test.py --full

# 7. Disable debug output
python test/Long_Memory_test.py --no-debug
```

### Expected Output

The script will:
1. Load test questions from the specified file
2. Process each conversation history and build memory
3. Answer each question using the memory system
4. Save results to the output file (one JSON object per line)

Results format:
```json
{
  "task_id": "task_001",
  "question": "What is my favorite color?",
  "predicted_answer": "Your favorite color is blue.",
  "ground_truth": "blue"
}
```

### Verify Results

```bash
# Check output file
ls -lh test/long_memory_results.json

# Count total questions processed
wc -l test/long_memory_results.json
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

## Troubleshooting

### Neo4j Connection Failed

```bash
# Check if Neo4j is running
neo4j status

# Start Neo4j
neo4j start

# Verify connection
python utils/connection_tests.py
```

### Embedding Server Issues

```bash
# If using local embedding, check server status
curl http://127.0.0.1:8000/health

# Alternative: Use online embedding API
# Edit .env:
GRAPHRAG_EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
GRAPHRAG_EMBEDDING_API_KEY=your-api-key
```

### Out of Memory

```bash
# Reduce batch size in .env
EMBED_BATCH_SIZE=1
EMBED_MAX_CONCURRENCY=1
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
