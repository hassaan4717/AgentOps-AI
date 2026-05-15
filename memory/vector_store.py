"""
Vector Store for Memory Retrieval

This module provides semantic search capabilities for memory retrieval using
vector embeddings and a local Chroma database.

=============================================================================
WHAT ARE EMBEDDINGS?
=============================================================================

Embeddings are numerical representations (vectors) of text that capture semantic meaning.

Example:
    Text: "Explain vector databases"
    Embedding: [0.23, -0.15, 0.87, ..., 0.34]  (1536 numbers)

Key Properties:
1. Similar text → Similar vectors (close together in vector space)
2. Captures meaning, not just keywords
3. Language models like Gemini produce embeddings as a "side effect" of understanding text

Think of embeddings as coordinates in a high-dimensional space where semantically
similar concepts are located near each other.

=============================================================================
WHY VECTORS ARE BETTER THAN KEYWORD MATCHING
=============================================================================

Keyword Matching (Current Implementation):
    Query: "Explain vector databases"
    Stored: "Describe embedding storage systems"
    Result: ❌ NO MATCH (no shared keywords)

Vector/Semantic Search (This Implementation):
    Query: "Explain vector databases"
    Stored: "Describe embedding storage systems"  
    Result: ✅ MATCH (semantically similar, ~85% similarity)

Advantages of Vector Search:
1. SEMANTIC UNDERSTANDING:
   - Matches by meaning, not just exact words
   - "car" matches "automobile", "vehicle", "sedan"
   
2. HANDLES SYNONYMS:
   - "fast" → "quick", "rapid", "speedy"
   - No need for manual synonym lists
   
3. MULTILINGUAL POTENTIAL:
   - Can match across languages (with multilingual models)
   - "Hello" matches "Bonjour", "Hola"
   
4. TYPO TOLERANCE:
   - Embeddings are robust to minor spelling errors
   - "databse" still close to "database"

5. CONTEXT AWARENESS:
   - "apple" (fruit) vs "apple" (company) have different embeddings
   - Understands context from surrounding words

Trade-offs:
- Slower: Requires computing embeddings (API call or local model)
- Complex: Needs vector database infrastructure
- Black Box: Harder to debug than keyword matching
- Cost: Embedding APIs have per-token costs

=============================================================================
WHY THIS IS ONLY USED FOR MEMORY RETRIEVAL
=============================================================================

This vector store is EXCLUSIVELY for finding relevant past task completions
during the Supervisor's planning phase. It is NOT used for:

❌ General knowledge retrieval (use RAG/document search instead)
❌ User query understanding (LLM handles that)
❌ Tool selection (agent logic handles that)
❌ Caching LLM outputs (different system entirely)

Why Memory-Only?
1. FOCUSED SCOPE:
   - Memory entries have consistent structure (user_goal, summary, output)
   - Purpose-built for "find similar past tasks"
   - Different from generic document search

2. HIGH QUALITY DATA:
   - Only stores validated outputs (score >= 8)
   - Clean, well-structured data ideal for embeddings
   - No noise from failed attempts or drafts

3. SMALL DATASET:
   - Max 100-1000 memory entries (manageable size)
   - Vector DB overhead justified by better retrieval
   - Would be overkill for 10-20 entries

4. CRITICAL USE CASE:
   - Supervisor planning benefits most from semantic matching
   - Finding similar past tasks prevents redundant research
   - Slight latency (embedding computation) is acceptable here

5. AVOIDS COMPLEXITY:
   - Keeps vector DB isolated to one use case
   - Easier to debug, test, and maintain
   - Clear boundaries prevent scope creep

=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

# ChromaDB: Local vector database (no server required)
# - Embedded mode: runs in-process, stores data in local directory
# - Fast for small-medium datasets (< 1M vectors)
# - Open source, no API keys needed
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

# Google's embedding model (compatible with Gemini outputs)
# We use Google's model because:
# 1. Free tier available
# 2. Same provider as our LLM (Gemini) = consistent ecosystem
# 3. High quality embeddings optimized for semantic search
# 4. 768-dimensional vectors (smaller than OpenAI = faster)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import memory record structure from existing memory module
from memory.memory_store import MemoryRecord, load_all_memories


# =============================================================================
# Configuration
# =============================================================================

# Directory where Chroma will store its database files
# - Uses SQLite under the hood for metadata
# - Stores vectors in efficient binary format
# - All data stays local (no cloud storage)
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Chroma collection name for memory entries
# Collections are like "tables" in traditional databases
# We use a single collection because all our data has the same structure
MEMORY_COLLECTION_NAME = "memory_records"

# Embedding model configuration
# gemini-embedding-001 is Google's embedding model for Gemini API
# - Works with the same API key as Gemini LLMs
# - Supports embedContent method for generating embeddings
# - Good quality for semantic search tasks
# Note: text-embedding-004 requires a different API (Vertex AI)
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"

# Distance metric for similarity search
# Options:
# - "cosine": Measures angle between vectors (most common for text)
# - "l2": Euclidean distance (better for numeric features)
# - "ip": Inner product (faster but less intuitive)
# We use cosine because it's standard for semantic similarity
DISTANCE_METRIC = "cosine"


# =============================================================================
# Vector Store Initialization
# =============================================================================


class VectorMemoryStore:
    """
    Vector-based memory retrieval using Chroma and Google embeddings.
    
    This class wraps Chroma DB to provide semantic search over memory records.
    It handles:
    - Embedding generation (user_goal → vector)
    - Vector storage and indexing
    - Similarity search (find most relevant memories)
    - Synchronization with JSON memory store
    
    Usage:
        store = VectorMemoryStore()
        store.sync_from_json()  # Load existing memories
        results = store.find_similar(user_goal="Explain vector databases", top_k=3)
    """
    
    def __init__(self):
        """
        Initialize the vector store with Chroma and Google embeddings.
        
        Steps:
        1. Create local directory for Chroma database
        2. Initialize Chroma client (embedded mode)
        3. Load or create collection for memory records
        4. Initialize Google embedding model
        """
        
        # Step 1: Ensure database directory exists
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Initialize Chroma client in embedded mode
        # Settings:
        # - persist_directory: Where to store database files
        # - anonymized_telemetry: Disable usage tracking (privacy)
        # - allow_reset: Enable collection deletion (useful for development)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Step 3: Get or create collection
        # Collections store vectors + metadata + documents
        # If collection exists, it's loaded; if not, it's created
        # 
        # Why we store metadata:
        # - Chroma doesn't just store vectors; it stores the full record
        # - Metadata = structured fields (score, created_at, etc.)
        # - Documents = the text we embedded (user_goal)
        # - IDs = unique identifiers (memory record IDs)
        self.collection = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={
                "hnsw:space": DISTANCE_METRIC,  # Distance function for search
                "description": "Memory records for AgentOps AI Platform",
            },
        )
        
        # Step 4: Initialize embedding model
        # This model converts text → vectors
        # Requires GOOGLE_API_KEY environment variable
        # 
        # The model is stateless (no training on our data)
        # Each text → vector conversion is independent
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google embeddings. "
                f"Ensure GOOGLE_API_KEY is set in environment. Error: {e}"
            )
    
    def sync_from_json(self) -> int:
        """
        Synchronize vector store with the JSON memory file.
        
        This function:
        1. Loads all memories from memory.json (ground truth)
        2. Compares with what's in the vector store
        3. Adds missing memories (generates embeddings + stores vectors)
        4. Removes deleted memories (cleanup)
        
        Returns:
            Number of memories added to the vector store
        
        Why Synchronization is Needed:
        - JSON file is the source of truth (simpler, more reliable)
        - Vector store is a secondary index for fast retrieval
        - If they get out of sync (crash, bug), this fixes it
        
        When to Call:
        - On startup (ensure vector store is up to date)
        - After bulk memory imports
        - After manual JSON edits
        - Periodically (e.g., daily) as a safety check
        """
        
        # Step 1: Load all memories from JSON
        all_memories = load_all_memories()
        
        if not all_memories:
            print("ℹ️  No memories found in JSON file. Vector store is empty.")
            return 0
        
        # Step 2: Get existing IDs in vector store
        # Chroma's get() returns all records if no filter is provided
        try:
            existing_records = self.collection.get()
            existing_ids = set(existing_records["ids"]) if existing_records["ids"] else set()
        except Exception as e:
            print(f"⚠️  Error reading existing vectors: {e}. Treating as empty.")
            existing_ids = set()
        
        # Step 3: Find memories that need to be added
        memories_to_add = [m for m in all_memories if m.id not in existing_ids]
        
        if not memories_to_add:
            print(f"✅ Vector store already in sync ({len(all_memories)} memories)")
            return 0
        
        # Step 4: Generate embeddings and add to vector store
        # We batch this for efficiency (one API call per memory)
        added_count = 0
        for memory in memories_to_add:
            try:
                self._add_memory(memory)
                added_count += 1
            except Exception as e:
                print(f"⚠️  Failed to add memory {memory.id}: {e}")
                # Continue with other memories (don't crash on one failure)
        
        print(f"✅ Synced {added_count} new memories to vector store")
        
        # Step 5: Remove memories that were deleted from JSON
        # (This handles the case where someone manually deleted from JSON)
        json_ids = {m.id for m in all_memories}
        ids_to_remove = existing_ids - json_ids
        
        if ids_to_remove:
            try:
                self.collection.delete(ids=list(ids_to_remove))
                print(f"🧹 Removed {len(ids_to_remove)} deleted memories from vector store")
            except Exception as e:
                print(f"⚠️  Error removing deleted memories: {e}")
        
        return added_count
    
    def _add_memory(self, memory: MemoryRecord) -> None:
        """
        Add a single memory record to the vector store.
        
        Steps:
        1. Combine user_goal + summary into a single text for embedding
        2. Generate embedding from combined text
        3. Store: vector + metadata + document + ID
        
        Why we embed BOTH user_goal AND summary:
        ==========================================
        1. USER_GOAL: Captures the intent/question
           - "Explain vector databases"
           - This is what new queries look like
        
        2. SUMMARY: Captures the solution/approach
           - "Vector databases use embeddings for semantic search..."
           - Provides context about HOW the problem was solved
        
        3. COMBINED: Maximizes retrieval accuracy
           - Goal matches similar questions
           - Summary matches similar solutions
           - Example: Query "Explain embeddings" matches:
             * Goal: "Explain vector databases" (related question)
             * Summary: "...uses embeddings for..." (solution mentions embeddings)
        
        Why NOT embed final_output:
        - Too long (2000 chars) = expensive embeddings
        - Contains implementation details, not concepts
        - Summary already distills the key points
        
        Args:
            memory: A MemoryRecord instance to add
        
        Raises:
            Exception: If embedding generation or storage fails
        """
        
        # Step 1: Combine user_goal and summary for richer semantic search
        # Format: "Goal: <goal>\nSummary: <summary>"
        # The labels help the model understand the structure
        combined_text = f"Goal: {memory.user_goal}\n\nSummary: {memory.summary}"
        
        # Step 2: Generate embedding from combined text
        # This is an API call to Google's embedding service
        # Returns a list of floats (768 dimensions for this model)
        # Cost: ~0.0001 cents per query (negligible)
        try:
            embedding = self.embeddings.embed_query(combined_text)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
        
        # Step 3: Add to Chroma collection
        # Chroma stores:
        # - ids: Unique identifier (memory.id) ← Used as document ID
        # - embeddings: The vector we just computed (from goal + summary)
        # - documents: The combined text we embedded (for reference)
        # - metadatas: Additional fields (user_goal, summary, score, etc.)
        #
        # Why use memory.id as document ID:
        # - Ensures uniqueness (no duplicate embeddings)
        # - Enables updates (re-embedding with same ID overwrites)
        # - Allows cross-referencing with JSON storage
        # - Facilitates debugging (ID appears in both stores)
        #
        # Why store combined_text as document:
        # - Shows what was actually embedded (transparency)
        # - Useful for debugging retrieval issues
        # - Helps understand why certain matches occurred
        #
        # Why store user_goal separately in metadata:
        # - Need to return just the goal (not goal+summary combo)
        # - Metadata preserves the original structure
        # - Allows reconstructing MemoryRecord without JSON read
        try:
            self.collection.add(
                ids=[memory.id],  # ← Memory ID as vector document ID
                embeddings=[embedding],
                documents=[combined_text],  # What we embedded (for reference)
                metadatas=[{
                    "user_goal": memory.user_goal,  # Store original fields
                    "summary": memory.summary,
                    "final_output": memory.final_output,
                    "score": memory.score,
                    "created_at": memory.created_at.isoformat(),
                }],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add to Chroma: {e}")
    
    def find_similar(
        self,
        user_goal: str,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Find the most semantically similar memories to a given user goal.
        
        This is the main retrieval function that replaces keyword matching.
        
        HOW EMBEDDING COMPARISON WORKS:
        ===============================
        Query side: We embed just the user_goal (the question being asked)
        Storage side: We embedded user_goal + summary (question + solution)
        
        This asymmetry is INTENTIONAL:
        - Query: "Explain vector databases" (user's question)
        - Stored: "Goal: Explain embeddings\n\nSummary: Embeddings are vectors..." (full context)
        - Match: Yes! Query about "vector databases" matches stored summary mentioning "embeddings"
        
        Benefits:
        1. Query matches similar QUESTIONS (same goal, different wording)
        2. Query matches similar SOLUTIONS (different goal, but same concepts in summary)
        3. Maximizes recall (finds more relevant memories)
        
        Example:
        - Query: "How to set up semantic search?"
        - Matches: 
          * Goal: "Explain vector databases" (different question, but summary has "semantic search")
          * Goal: "Configure Chroma for retrieval" (similar task)
        
        Steps:
        1. Embed the query (user_goal → vector)
        2. Search Chroma for nearest neighbors (similar vectors)
        3. Filter by similarity threshold
        4. Convert results back to MemoryRecord objects
        
        Args:
            user_goal: The current user request to match against stored memories
            top_k: Maximum number of results to return (default: 3)
            min_similarity: Minimum cosine similarity (0-1) to include (default: 0.6)
                           - 0.9-1.0: Nearly identical (exact rephrasing)
                           - 0.7-0.9: Highly similar (same topic, different wording)
                           - 0.5-0.7: Somewhat similar (related topics)
                           - < 0.5: Weakly related or unrelated
        
        Returns:
            List of (MemoryRecord, similarity_score) tuples, sorted by similarity
            Empty list if no good matches found
        
        Example:
            results = store.find_similar("Explain vector databases", top_k=3)
            for memory, score in results:
                print(f"{score:.2f}: {memory.user_goal}")
            # Output:
            # 0.92: Explain vector databases for RAG
            # 0.78: Describe embedding storage systems (summary mentions vectors!)
            # 0.65: What are semantic search tools?
        """
        
        if not user_goal or not user_goal.strip():
            return []
        
        # Step 1: Generate embedding for query
        try:
            query_embedding = self.embeddings.embed_query(user_goal)
        except Exception as e:
            print(f"⚠️  Failed to generate query embedding: {e}")
            return []
        
        # Step 2: Search Chroma for nearest neighbors
        # Chroma's query() returns:
        # - ids: List of matching record IDs
        # - distances: Distance between query and each result
        # - documents: The original text (user_goal)
        # - metadatas: Additional fields we stored
        #
        # Distance to Similarity Conversion:
        # - Cosine distance (Chroma) = 1 - cosine similarity
        # - So: similarity = 1 - distance
        # - distance=0.1 → similarity=0.9 (very similar)
        # - distance=0.9 → similarity=0.1 (barely similar)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"⚠️  Vector search failed: {e}")
            return []
        
        # Step 3: Convert results to MemoryRecord objects
        # Chroma returns nested lists (supports batch queries)
        # We only sent one query, so we unpack the first result
        if not results["ids"] or not results["ids"][0]:
            # No results found
            return []
        
        matched_memories: list[tuple[MemoryRecord, float]] = []
        
        for idx, record_id in enumerate(results["ids"][0]):
            # Convert distance to similarity
            distance = results["distances"][0][idx]
            similarity = 1.0 - distance
            
            # Filter by minimum similarity threshold
            if similarity < min_similarity:
                continue
            
            # Reconstruct MemoryRecord from metadata
            # (This avoids reading from JSON again)
            # Note: user_goal is now in metadata (not documents, which has combined text)
            metadata = results["metadatas"][0][idx]
            try:
                from datetime import datetime
                memory = MemoryRecord(
                    id=record_id,
                    user_goal=metadata["user_goal"],  # ← Now in metadata
                    summary=metadata["summary"],
                    final_output=metadata["final_output"],
                    score=metadata["score"],
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                )
                matched_memories.append((memory, similarity))
            except Exception as e:
                print(f"⚠️  Failed to reconstruct memory {record_id}: {e}")
                continue
        
        # Results are already sorted by similarity (Chroma returns nearest first)
        return matched_memories
    
    def add_new_memory(self, memory: MemoryRecord) -> bool:
        """
        Add a newly created memory to the vector store.
        
        This should be called after save_memory() succeeds in memory_store.py.
        
        Args:
            memory: The MemoryRecord that was just saved to JSON
        
        Returns:
            True if added successfully, False otherwise
        
        Usage Pattern:
            # In your memory write logic:
            if save_memory(record):  # Save to JSON first
                vector_store.add_new_memory(record)  # Then add to vector store
        """
        try:
            self._add_memory(memory)
            return True
        except Exception as e:
            print(f"⚠️  Failed to add memory to vector store: {e}")
            return False
    
    def clear_all(self) -> None:
        """
        Delete all vectors from the collection.
        
        ⚠️  DANGEROUS: This does not delete from JSON, only from vector store.
        Use this for testing or when rebuilding the index.
        
        To fully reset memory:
        1. Delete memory.json
        2. Call this method
        """
        try:
            self.client.delete_collection(MEMORY_COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=MEMORY_COLLECTION_NAME,
                metadata={"hnsw:space": DISTANCE_METRIC},
            )
            print("🧹 Cleared all vectors from memory store")
        except Exception as e:
            print(f"⚠️  Failed to clear vector store: {e}")


# =============================================================================
# Usage Example
# =============================================================================
#
# from memory.memory_store import MemoryRecord, save_memory
# from memory.vector_store import VectorMemoryStore
# from datetime import datetime, timezone
#
# # EXAMPLE 1: Saving a new memory (automatic vector store integration)
# # ====================================================================
# new_memory = MemoryRecord(
#     id="123",
#     user_goal="Explain vector databases for semantic search",
#     summary="Vector databases store embeddings and support similarity search...",
#     final_output="Full explanation here...",
#     score=9,
#     created_at=datetime.now(timezone.utc),
# )
#
# # Save to JSON - vector store update happens automatically!
# if save_memory(new_memory):
#     print("✅ Memory saved to JSON and vector store")
# # Behind the scenes:
# # 1. JSON write succeeds
# # 2. Embedding generated from: user_goal + summary
# # 3. Vector stored with memory.id as document ID
# # 4. If vector storage fails, JSON still has it (non-blocking)
#
#
# # EXAMPLE 2: Searching for similar memories
# # ==========================================
# store = VectorMemoryStore()
#
# # Search by user goal - matches both similar goals AND similar summaries
# results = store.find_similar(
#     user_goal="How to use embeddings?",  # Query
#     top_k=3,
#     min_similarity=0.6,
# )
#
# print(f"Found {len(results)} similar memories:")
# for memory, similarity in results:
#     print(f"\n[Similarity: {similarity:.2%}]")
#     print(f"Goal: {memory.user_goal}")
#     print(f"Summary: {memory.summary[:100]}...")
#     print(f"Score: {memory.score}/10")
#
# # Example output:
# # [92%] Goal: Explain vector databases for semantic search
# #       Summary: Vector databases store embeddings and support similarity search...
# #       (High match: query "embeddings" matches summary "embeddings")
#
#
# # EXAMPLE 3: Syncing vector store with existing JSON memories
# # =============================================================
# # Run this on startup or if vector store gets out of sync
# store = VectorMemoryStore()
# added_count = store.sync_from_json()
# print(f"✅ Synced {added_count} memories to vector store")
#
# =============================================================================


# =============================================================================
# Performance Notes
# =============================================================================
#
# SPEED:
# - Embedding generation: ~50-200ms per query (API call to Google)
# - Vector search (local): <10ms for 100-1000 entries
# - Total latency: ~100-250ms (acceptable for planning phase)
#
# ACCURACY:
# - Vector search typically outperforms keyword matching by 20-40%
# - Recall (finding relevant results) is much higher
# - Precision (avoiding false positives) is similar
#
# SCALING:
# - This setup works well for 100-10,000 memory entries
# - Beyond 10k entries, consider:
#   - Cloud vector DB (Pinecone, Weaviate)
#   - Distributed search (Elasticsearch with vector plugin)
#   - Hierarchical indexing (coarse + fine search)
#
# COST:
# - Chroma: Free (local, open source)
# - Google embeddings: Free tier (15k queries/min)
# - After free tier: ~$0.0001 per query (very cheap)
#
# =============================================================================
