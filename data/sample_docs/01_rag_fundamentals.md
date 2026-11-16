# RAG Fundamentals: Architecture, Embeddings, and Grounding

## 1. Why Large Language Models Hallucinate
Large Language Models (LLMs) are trained to predict the next token based on statistical patterns learned during pre-training. While this makes them fluent writers and capable reasoners, parametric memory alone has three critical limitations in production systems:
1. **Knowledge Cutoff**: An LLM cannot know about events, software releases, or policy updates that occurred after its training cutoff date.
2. **Private & Proprietary Data**: Your organization's internal wiki, customer tickets, and local codebases were never in the model's public training set.
3. **Confident Hallucination**: When asked about a specific fact it does not know (such as internal protocol `PROTO-RAG-101`), an ungrounded LLM will often fabricate a plausible-sounding but false explanation rather than admitting uncertainty.

## 2. The Core RAG Pipeline (Retrieve, Augment, Generate)
Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), solves these limitations by connecting the LLM to an external, verifiable knowledge base at query time. Every RAG system operates in two main phases:

### Phase A: Offline Indexing
- **Ingestion**: Load raw documents (Markdown, PDFs, HTML, database rows).
- **Chunking**: Split long documents into smaller, semantically coherent passages (typically 200 to 600 tokens) with a small overlap (10% to 20%) so sentences on boundaries do not lose context.
- **Embedding**: Pass each text chunk through an **Embedding Model** (such as `bge-base-en-v1.5`, which outputs a 768-dimensional float vector) to capture its semantic meaning in high-dimensional space.
- **Storage**: Save both the raw text chunk and its embedding vector into a searchable index or database (such as a Python list for tiny demos, or **DuckDB** for persistent local workloads).

### Phase B: Online Query & Generation
- **Query Embedding**: Convert the user's incoming question into a vector using the *exact same* embedding model used during indexing.
- **Similarity Retrieval**: Compute the **Cosine Similarity** between the query vector and all stored chunk vectors, selecting the top $k$ highest-scoring chunks.
- **Prompt Augmentation**: Construct a grounded prompt that places the retrieved chunks inside a clear context block and instructs the LLM to answer *only* using the provided evidence.
- **Generation**: Send the augmented prompt to a local LLM (via `Llama.app` on `http://localhost:9931/v1`) to synthesize a cited, factual response.

## 3. Mathematical Intuition: Cosine Similarity
Why do we use cosine similarity instead of Euclidean distance? Cosine similarity measures the cosine of the angle $\theta$ between two vectors $A$ and $B$:
$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$
Because it normalizes by vector magnitude ($\|A\|$ and $\|B\|$), two passages covering the same topic with different lengths will still point in nearly the same direction in embedding space, yielding a similarity score close to `1.0`. Orthogonal (unrelated) topics score near `0.0`.
