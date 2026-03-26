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

### Step 1: Download Dataset

```bash
# Clone LongMemEval repository
git clone https://github.com/xiaowu0162/LongMemEval.git

# Copy test files to your project
cp LongMemEval/data/sampled_test_questions.json data/long_memory_eval/
cp LongMemEval/data/medium_test_questions.json data/long_memory_eval/
```

### Step 2: Verify Directory Structure

```
data/
├── long_memory_eval/
│   ├── sampled_test_questions.json    # Sample setting (required)
│   └── medium_test_questions.json     # Hard setting (required)
├── books/                              # Character profiles (optional)
└── world_knowledge/                    # World knowledge base (optional)
```

### Step 3: Verify Data Files

```bash
# Check if files exist
ls -lh data/long_memory_eval/

# Expected output:
# sampled_test_questions.json  (~2-5 MB)
# medium_test_questions.json   (~10-20 MB)
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

## Running Experiments

### Quick Start: Sample Setting

```bash
# 1. Ensure Neo4j is running
neo4j status

# 2. Clear previous data (optional)
python utils/clear_long_term_memory.py

# 3. Run evaluation on sample setting
python test/Long_Memory_test.py

# 4. Check results
cat result/simple/long_memory_results.jsonl
```

### Full Reproduction Steps

#### Step 1: Verify Environment

```bash
# Test Neo4j connection
python utils/connection_tests.py

# Expected output:
# ✓ Neo4j connection successful
# ✓ LLM API connection successful
# ✓ Embedding model loaded
```

#### Step 2: Initialize Database

```bash
# Create Neo4j schema and indexes
python utils/init_neo4j_schema.py
python utils/create_fulltext_index.py

# Expected output:
# ✓ Created constraints and indexes
# ✓ Fulltext index created
```

#### Step 3: Start Embedding Server (if using local)

```bash
# Terminal 1: Start embedding server
python embedding_server.py

# Expected output:
# INFO: Uvicorn running on http://127.0.0.1:8000
# Model loaded: BAAI/bge-m3
```

#### Step 4: Run Evaluation

```bash
# Terminal 2: Run test
python test/Long_Memory_test.py

# Command line options:
# --data_path: Path to test questions (default: data/long_memory_eval/sampled_test_questions.json)
# --output_path: Path to save results (default: result/simple/long_memory_results.jsonl)
# --clear: Clear database before running (default: False)

# Example with options:
python test/Long_Memory_test.py \
    --data_path data/long_memory_eval/medium_test_questions.json \
    --output_path result/medium/long_memory_results.jsonl \
    --clear
```

#### Step 5: Verify Results

```bash
# Check output file
ls -lh result/simple/long_memory_results.jsonl

# Count total questions
wc -l result/simple/long_memory_results.jsonl

# View sample results
head -n 5 result/simple/long_memory_results.jsonl | jq .
```

### Expected Output Format

Each line in the result file is a JSON object:

```json
{
  "task_id": "task_001",
  "question": "What is my favorite color?",
  "predicted_answer": "Your favorite color is blue.",
  "ground_truth": "blue",
  "evidence_ids": [1, 5, 12],
  "retrieval_time": 0.234,
  "generation_time": 1.567
}
```

### Running Ablation Experiments

```bash
# Disable structured memory
export DISABLE_STRUCTURED_MEMORY=1
python test/Long_Memory_test.py --output_path result/ablation/no_structured.jsonl

# Disable RAW channel
export DISABLE_RAW_CHANNEL=1
python test/Long_Memory_test.py --output_path result/ablation/no_raw.jsonl

# Disable query expansion
export DISABLE_QUERY_EXPAND=1
python test/Long_Memory_test.py --output_path result/ablation/no_expand.jsonl

# Disable temporal modeling
export DISABLE_TEMPORAL_MODEL=1
python test/Long_Memory_test.py --output_path result/ablation/no_temporal.jsonl
```

See `Appendix_DuMF_Agent.tex` for complete ablation configuration details.

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

### Common Issues

#### 1. Neo4j Connection Failed

```bash
# Check if Neo4j is running
neo4j status

# Start Neo4j
neo4j start

# Check connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('neo4j://localhost:7687', auth=('neo4j', 'your-password')); driver.verify_connectivity(); print('✓ Connected')"
```

#### 2. Embedding Server Not Responding

```bash
# Check if server is running
curl http://127.0.0.1:8000/health

# Restart server
pkill -f embedding_server.py
python embedding_server.py
```

#### 3. Out of Memory

```bash
# Reduce batch size in .env
EMBED_BATCH_SIZE=1
EMBED_MAX_CONCURRENCY=1

# Or use online embedding API instead of local
GRAPHRAG_EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
GRAPHRAG_EMBEDDING_API_KEY=your-api-key
```

#### 4. API Rate Limit

```bash
# Use multiple API keys for load balancing
CHEAP_GRAPHRAG_CHAT_API_KEY_A=sk-key-1
CHEAP_GRAPHRAG_CHAT_API_KEY_B=sk-key-2
```

#### 5. Test Data Not Found

```bash
# Verify data files exist
ls -lh data/long_memory_eval/

# If missing, download from LongMemEval repository
git clone https://github.com/xiaowu0162/LongMemEval.git
cp LongMemEval/data/*.json data/long_memory_eval/
```

### Performance Optimization

#### For Faster Evaluation

1. Use online embedding API (faster than local on CPU)
2. Increase `EMBED_MAX_CONCURRENCY` if using remote API
3. Use SSD for Neo4j database storage
4. Allocate more memory to Neo4j (edit `neo4j.conf`)

#### For Lower Cost

1. Use local embedding model (free but slower)
2. Use cheaper LLM models (e.g., gpt-3.5-turbo)
3. Reduce retrieval top-k values in `config.py`

## Evaluation Results

Performance on LongMemEval benchmark:

| Method | Overall Acc. (Sample) | Overall Acc. (Hard) |
|--------|----------------------|---------------------|
| LLM    | 0.7500               | 0.4138              |
| RAG    | 0.6724               | 0.5172              |
| Mem0   | 0.5690               | 0.2759              |
| GA     | 0.5862               | 0.2414              |
| **DuMF-Agent** | **0.7241**   | **0.5517**          |

### Reproducing Results

To reproduce the results in the paper:

```bash
# 1. Sample setting
python test/Long_Memory_test.py \
    --data_path data/long_memory_eval/sampled_test_questions.json \
    --output_path result/simple/long_memory_results.jsonl

# 2. Hard setting
python test/Long_Memory_test.py \
    --data_path data/long_memory_eval/medium_test_questions.json \
    --output_path result/medium/long_memory_results.jsonl

# 3. Calculate metrics (if evaluation script available)
python evaluate_results.py result/simple/long_memory_results.jsonl
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LongMemEval benchmark for evaluation framework
- Neo4j for graph database support
