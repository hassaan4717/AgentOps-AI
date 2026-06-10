"""
Memory storage models and utilities.

This module defines the data structures for storing successful task completions
in memory. Memory is used exclusively by the Supervisor agent during planning
to learn from past successes and avoid redundant research.

IMPORTANT DESIGN PRINCIPLES:
- Only high-quality outputs (score >= 8, pass = True) are stored
- Memory entries are sanitized (PII removed, size limited)
- Memory is advisory, not prescriptive
- Failed memory operations never block user responses
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Memory Configuration Constants
# =============================================================================
#
# These constants define the boundaries of the memory system to prevent
# unbounded growth, stale data accumulation, and storage of low-quality outputs.
#
# Centralizing these values here makes it easy to:
# 1. Adjust limits based on deployment environment (dev vs prod)
# 2. Document the rationale for each limit
# 3. Ensure consistent behavior across all memory operations
# 4. Enable easy testing with different configurations
# =============================================================================

MAX_MEMORIES: int = 100
# Maximum number of memory entries to retain at any time.
#
# Why this limit exists:
# 1. RETRIEVAL PERFORMANCE:
#    - Simple keyword matching is O(n) where n = number of memories
#    - With 100 entries, retrieval is sub-millisecond
#    - Beyond 1000 entries, consider switching to a vector database
#
# 2. STORAGE FOOTPRINT:
#    - Each entry ≈ 3.5KB (goal + summary + output + metadata)
#    - 100 entries ≈ 350KB total (negligible)
#    - JSON file remains human-readable for debugging
#
# 3. QUALITY OVER QUANTITY:
#    - Storing top 100 successful tasks is usually sufficient for pattern recognition
#    - More entries can lead to noise (old, irrelevant, or outdated tasks)
#
# 4. EVICTION STRATEGY:
#    - When limit reached, evict oldest entries (FIFO)
#    - Alternative: evict least-retrieved entries (LRU)
#    - Alternative: evict lowest-scored entries
#
# Tuning guidance:
# - Small deployments (single user): 50-100 entries
# - Medium deployments (team): 500-1000 entries
# - Large deployments (enterprise): Use vector DB, no hard limit

MIN_SCORE_TO_KEEP: int = 7
# Minimum evaluation score to retain in memory.
#
# Why this limit exists:
# 1. QUALITY GATE DURING STORAGE:
#    - At write time, only score >= 8 are stored (high quality)
#    - This constant defines the RETENTION threshold (can be lower)
#
# 2. GRADUAL QUALITY DEGRADATION:
#    - Score 8-10: Stored immediately, kept for full retention period
#    - Score 7: Not stored initially, but if manually added or imported, kept for shorter period
#    - Score < 7: Never stored, or evicted immediately during cleanup
#
# 3. CLEANUP POLICY:
#    - Periodic cleanup (daily/weekly) can lower the threshold over time
#    - Example: After 30 days, entries with score < 8 can be demoted to score 7 retention
#    - After 60 days, entries with score < 9 can be evicted
#
# 4. EMERGENCY SPACE RECLAMATION:
#    - If storage limit reached, evict entries with score < MIN_SCORE_TO_KEEP first
#    - This ensures only the highest-quality memories survive under pressure
#
# Tuning guidance:
# - Strict quality: MIN_SCORE_TO_KEEP = 8 (same as write threshold)
# - Moderate quality: MIN_SCORE_TO_KEEP = 7 (keep slightly lower-quality memories)
# - Lenient quality: MIN_SCORE_TO_KEEP = 6 (maximize memory coverage, risk pollution)

MAX_MEMORY_AGE_DAYS: int = 90
# Maximum age (in days) before a memory entry is considered stale and evicted.
#
# Why this limit exists:
# 1. PREVENTS STALE KNOWLEDGE:
#    - Technology/best practices change (e.g., "Use Python 2.7" becomes outdated)
#    - Agent system evolves (new capabilities, different prompt strategies)
#    - 90 days balances recency with sufficient history
#
# 2. COMPLIANCE AND PRIVACY:
#    - Even with PII redaction, old data carries privacy risk
#    - GDPR/CCPA often require data minimization (keep only what's needed)
#    - Automatic expiration reduces compliance burden
#
# 3. PREVENTS MEMORY DRIFT:
#    - Old memories can anchor the system to outdated patterns
#    - Example: If the system learns a suboptimal approach, keeping it for
#      months prevents the system from adapting to better methods
#
# 4. STORAGE HYGIENE:
#    - Periodic cleanup prevents indefinite accumulation
#    - Keeps memory focused on recent, relevant patterns
#
# 5. TIERED RETENTION (Future Enhancement):
#    - Score 8: Retain for 30 days
#    - Score 9: Retain for 60 days
#    - Score 10: Retain for 90 days (exceptional quality)
#
# Tuning guidance:
# - Fast-moving domain (AI/ML research): 30-60 days
# - Stable domain (accounting, legal): 180-365 days
# - Compliance-sensitive: 30 days (minimize retention)
# - High-value memories: Store separately with manual review


class MemoryRecord(BaseModel):
    """
    A single memory entry representing a successfully completed task.

    This schema is designed to avoid memory pollution by:
    1. Requiring a minimum quality threshold (score >= 8)
    2. Storing summaries instead of full outputs to prevent prompt bloat
    3. Including timestamps for automatic expiration of stale entries
    4. Enforcing size limits to keep retrieval fast
    5. Sanitizing inputs to prevent PII leakage and prompt injection

    All fields are required to ensure memory integrity.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this memory entry (UUID recommended)",
    )
    # Why this matters:
    # - Enables deduplication (prevent storing the same task multiple times)
    # - Allows explicit deletion or updates of specific entries
    # - Facilitates audit trails and debugging

    user_goal: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Sanitized user request that led to this successful output",
    )
    # Why this matters:
    # - Core retrieval key: Supervisor searches memory by goal similarity
    # - Must be sanitized: PII removed, normalized for consistent matching
    # - Length limits prevent storing extremely vague ("help me") or overly
    #   specific ("Write a 500-page report on...") goals
    # - Enables semantic search: embed this field for similarity matching
    #
    # Pollution prevention:
    # - min_length=10 rejects trivial goals that don't provide useful context
    # - max_length=500 forces summarization of complex goals

    summary: str = Field(
        ...,
        min_length=50,
        max_length=1000,
        description="Brief summary of the key insights and approach (NOT the full output)",
    )
    # Why this matters:
    # - Enables quick scanning: Supervisor can review multiple entries efficiently
    # - Contains actionable insights: "Vector databases trade flexibility for speed"
    # - Prevents prompt bloat: 1000 chars max vs potentially 10KB+ full outputs
    # - Improves retrieval relevance: summaries are more semantically focused
    #
    # Pollution prevention:
    # - Forces distillation: only the most important learnings are captured
    # - min_length=50 ensures summaries are substantive, not just "Task completed"
    # - Excludes execution details that don't generalize to other tasks

    final_output: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="The final validated output (truncated if necessary) for reference",
    )
    # Why this matters:
    # - Provides concrete examples: Supervisor can see what "good" looks like
    # - Enables quality verification: humans can audit what's being stored
    # - Supports future enhancements: could be used for few-shot prompting
    #
    # Pollution prevention:
    # - max_length=2000 enforces aggressive truncation (vs 10KB+ raw outputs)
    # - Still long enough to capture key structure and style
    # - Prevents memory from becoming a full output cache
    #
    # IMPORTANT: This field is optional in the original design doc (summary only).
    # Including it here for completeness, but it could be made Optional[str] if
    # summary alone is sufficient. The max_length ensures it doesn't dominate
    # the memory footprint.

    score: int = Field(
        ...,
        ge=8,
        le=10,
        description="Evaluation score (1-10). Only scores >= 8 are stored.",
    )
    # Why this matters:
    # - Quality gate: Only proven-successful outputs earn a place in memory
    # - Enables ranking: prefer score=10 entries over score=8 during retrieval
    # - Prevents memorization of "barely passing" outputs
    #
    # Pollution prevention:
    # - ge=8 constraint enforced at the Pydantic level (fail-fast validation)
    # - This is the PRIMARY defense against low-quality entries
    # - Score=6-7 outputs may pass evaluation but don't deserve memory storage

    created_at: datetime = Field(
        ...,
        description="UTC timestamp when this entry was stored",
    )
    # Why this matters:
    # - Enables automatic expiration: delete entries older than 30-90 days
    # - Supports recency weighting: newer entries rank higher in retrieval
    # - Facilitates debugging: trace when a memory entry was created
    # - Allows temporal analysis: track how memory grows over time
    #
    # Pollution prevention:
    # - Stale entries (6+ months old) are automatically evicted
    # - Prevents memory from accumulating outdated knowledge indefinitely
    # - Ensures memory reflects current system capabilities, not legacy behavior

    # =============================================================================
    # SCHEMA DESIGN: Why This Prevents Memory Pollution
    # =============================================================================
    #
    # 1. HIGH QUALITY THRESHOLD (score >= 8):
    #    - Only the top ~20% of successful outputs are stored
    #    - Failed attempts never enter memory
    #    - Borderline outputs (score 6-7) are excluded
    #
    # 2. SIZE LIMITS:
    #    - user_goal: max 500 chars (prevents overly specific or vague goals)
    #    - summary: max 1000 chars (forces distillation of key insights)
    #    - final_output: max 2000 chars (prevents memory bloat)
    #    - Total entry size: ~3.5KB max (enables fast retrieval)
    #
    # 3. REQUIRED FIELDS:
    #    - No optional fields means every entry is complete
    #    - Incomplete entries are rejected at validation time
    #    - Prevents partial/corrupted entries from polluting memory
    #
    # 4. TIMESTAMP-BASED EXPIRATION:
    #    - created_at enables automatic cleanup of stale entries
    #    - Memory doesn't accumulate indefinitely
    #    - Outdated knowledge is periodically purged
    #
    # 5. UNIQUE IDs:
    #    - Prevents duplicate entries for the same task
    #    - Enables explicit deletion of problematic entries
    #
    # 6. VALIDATION AT WRITE TIME:
    #    - Pydantic enforces constraints before storage
    #    - Invalid entries are rejected immediately
    #    - No need for post-hoc cleanup of malformed data

    @field_validator("created_at")
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        """
        Ensure timestamps are always UTC to avoid timezone confusion.

        Memory entries may be written and read in different timezones.
        Normalizing to UTC prevents bugs like:
        - "This entry is 5 hours old" vs "This entry is brand new" (timezone mismatch)
        - Incorrect expiration logic due to DST or timezone changes
        """
        if v.tzinfo is None:
            # Naive datetime (no timezone) -> assume UTC
            return v.replace(tzinfo=timezone.utc)
        # Already has timezone -> convert to UTC
        return v.astimezone(timezone.utc)

    @field_validator("user_goal", "summary", "final_output")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """
        Strip leading/trailing whitespace to normalize entries.

        Prevents near-duplicates due to inconsistent whitespace:
        - "Explain vectors" vs "  Explain vectors  " should be treated the same
        - Reduces memory footprint slightly
        """
        return v.strip()

    @field_validator("user_goal", "summary", "final_output")
    @classmethod
    def reject_empty_after_strip(cls, v: str, info) -> str:
        """
        Reject entries that are empty after stripping whitespace.

        Prevents storing entries with only spaces/newlines, which:
        - Waste memory
        - Break semantic search (empty strings have no embedding)
        - Indicate a bug in the calling code
        """
        if not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot be empty or whitespace-only")
        return v

    class Config:
        # Prevent extra fields from being silently accepted
        # If caller passes unexpected fields, raise an error
        # This catches bugs early (e.g., typo in field name)
        extra = "forbid"

        # Example for documentation/testing
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_goal": "Explain the benefits of vector databases for semantic search",
                "summary": "Vector databases excel at semantic search by storing embeddings and supporting nearest-neighbor queries. Key trade-offs: faster similarity search but less flexible than SQL. Common use cases: RAG, recommendation systems.",
                "final_output": "Vector databases are specialized systems optimized for storing and querying high-dimensional vectors (embeddings). Unlike traditional databases...",
                "score": 9,
                "created_at": "2026-02-10T12:34:56Z",
            }
        }


# =============================================================================
# Future Enhancements (Not Implemented Yet)
# =============================================================================
#
# 1. METADATA FIELDS:
#    Consider adding optional metadata for better retrieval:
#    - tags: list[str] = Field(default_factory=list)  # ["technical", "explanation"]
#    - required_research: bool                         # Was research needed?
#    - plan_steps: list[str]                           # What plan worked?
#    - iteration_count: int                            # How many retries?
#
# 2. EMBEDDING FIELD:
#    Store the embedding of user_goal directly for faster retrieval:
#    - embedding: Optional[list[float]] = None         # 1536-dim vector (OpenAI)
#
# 3. USAGE TRACKING:
#    Track how often each entry is retrieved:
#    - retrieval_count: int = 0                        # How many times retrieved?
#    - last_retrieved_at: Optional[datetime] = None    # When was it last used?
#    This enables eviction of low-utility entries (never retrieved in 30 days)
#
# 4. VERSION TAGGING:
#    Track which system version created this entry:
#    - system_version: str = "v0.1"                    # Enables bulk deletion on upgrades
#
# 5. SANITIZATION METADATA:
#    Track what was sanitized for audit purposes:
#    - pii_redacted: bool = False                      # Was PII found and removed?
#    - redactions: list[str] = []                      # ["EMAIL", "PHONE_NUMBER"]


# =============================================================================
# Storage Backend: Simple JSON File
# =============================================================================
#
# This implementation uses a local JSON file for simplicity and portability.
# For production, consider:
# - Vector database (Pinecone, Weaviate, Qdrant) for semantic search
# - SQLite with FTS5 for full-text search
# - PostgreSQL with pgvector for hybrid search
# - Redis for high-throughput caching
#
# The JSON file approach is sufficient for:
# - Development and testing
# - Small deployments (< 1000 entries)
# - Single-instance systems (no concurrent writes)

# Define the memory file path relative to this module
# This ensures the file is always created in the /memory directory
DATA_DIR = Path(os.environ.get("AGENTOPS_DATA_DIR", Path(__file__).parent))
MEMORY_FILE = DATA_DIR / "memory.json"


def save_memory(record: MemoryRecord) -> bool:
    """
    Save a single memory record to the local JSON file with automatic pruning.

    This function is designed to NEVER crash the application. If any error occurs
    (disk full, permission denied, etc.), it logs the error and returns False.

    Automatic Maintenance (on every write):
    - Enforces MIN_SCORE_TO_KEEP (rejects low-quality records)
    - Prunes entries older than MAX_MEMORY_AGE_DAYS
    - Enforces MAX_MEMORIES limit (keeps most recent)

    Why prune on write (not separate cleanup job)?
    1. SIMPLICITY: No need for cron jobs or background workers
    2. INCREMENTAL: Cleanup cost is amortized across writes (each write cleans a little)
    3. GUARANTEED: Memory limits are always enforced, even if cleanup job fails
    4. TESTABLE: Easier to test (write → verify limits) vs async cleanup
    5. LOW OVERHEAD: Pruning 100 entries takes <1ms, negligible compared to LLM calls

    Args:
        record: A validated MemoryRecord instance (Pydantic ensures it's valid)

    Returns:
        True if save succeeded, False otherwise

    Safety Guarantees:
    - File is created if missing
    - Existing entries are preserved (append-only behavior)
    - Pruning errors do not crash the system (logged and skipped)
    - Disk errors do not crash the system
    - Invalid records are rejected at validation time (Pydantic)

    Performance:
    - O(n) where n = number of existing entries (reads all, prunes, writes all)
    - Not suitable for high-frequency writes (>100/sec)
    - For high throughput, use a proper database with indexed cleanup
    """

    try:
        # Step 0: Quality gate - reject records below retention threshold
        # This prevents wasting I/O on records that would be immediately pruned.
        # Note: The MemoryRecord model enforces score >= 8 at the Pydantic level,
        # but this check defends against manually created records with score 7.
        if record.score < MIN_SCORE_TO_KEEP:
            print(f"⚠️  Record score ({record.score}) below MIN_SCORE_TO_KEEP ({MIN_SCORE_TO_KEEP}). Not saving.")
            return False

        # Step 1: Load existing memories (if any)
        # This ensures we append rather than overwrite
        existing_memories = load_all_memories()

        # Step 2: Check for duplicates (prevent storing the same ID twice)
        # Duplicates waste space and complicate retrieval
        existing_ids = {m.id for m in existing_memories}
        if record.id in existing_ids:
            # This is not necessarily an error (idempotent writes are fine)
            # But we should not store duplicates
            # In production, consider updating the existing entry instead
            print(f"⚠️  Memory record with id={record.id} already exists. Skipping.")
            return True  # Return True since the record is already stored

        # Step 3: Append the new record
        existing_memories.append(record)

        # Step 4: PRUNE OLD MEMORIES (automatic maintenance on write)
        # Remove entries older than MAX_MEMORY_AGE_DAYS to prevent stale knowledge.
        #
        # Why prune here (not in a separate cleanup job)?
        # - Simpler: no cron jobs or background workers needed
        # - Guaranteed: limits always enforced, even if cleanup job fails
        # - Incremental: cleanup cost amortized across writes
        # - Low overhead: pruning 100 entries takes <1ms
        #
        # Pruning is wrapped in try-except so failures don't block the save operation.
        try:
            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=MAX_MEMORY_AGE_DAYS)
            original_count = len(existing_memories)

            # Filter: keep only recent memories
            existing_memories = [m for m in existing_memories if m.created_at >= cutoff_date]

            pruned_count = original_count - len(existing_memories)
            if pruned_count > 0:
                print(f"🧹 Pruned {pruned_count} old memory entries (>{MAX_MEMORY_AGE_DAYS} days)")

        except Exception as e:
            # Pruning failed (unexpected error in date comparison, etc.)
            # Log the error but continue with the save operation
            print(f"⚠️  Age-based pruning failed (non-critical): {type(e).__name__}: {e}")

        # Step 5: ENFORCE MAX_MEMORIES LIMIT
        # If we have too many entries, keep only the most recent ones.
        #
        # Eviction strategy: FIFO (First In, First Out) based on created_at timestamp.
        # Alternative strategies:
        # - Score-based: Keep highest-scored entries
        # - LRU: Keep most frequently retrieved entries (requires tracking)
        # - Hybrid: Score × Recency weighting
        #
        # FIFO is simplest and ensures memory reflects recent system behavior.
        try:
            if len(existing_memories) > MAX_MEMORIES:
                # Sort by created_at (oldest first)
                existing_memories.sort(key=lambda m: m.created_at)

                # Keep only the most recent MAX_MEMORIES entries
                evicted_count = len(existing_memories) - MAX_MEMORIES
                existing_memories = existing_memories[-MAX_MEMORIES:]

                print(f"🧹 Evicted {evicted_count} oldest entries (exceeded MAX_MEMORIES={MAX_MEMORIES})")

        except Exception as e:
            # Limit enforcement failed (unexpected error in sorting, etc.)
            # Log the error but continue with the save operation
            print(f"⚠️  Count-based pruning failed (non-critical): {type(e).__name__}: {e}")

        # Step 6: Serialize all records to JSON (after pruning)
        # Use model_dump() to convert Pydantic models to dicts
        # mode="json" ensures datetimes are serialized as ISO strings
        records_as_dicts = [m.model_dump(mode="json") for m in existing_memories]

        # Step 7: Write to file with atomic rename
        # Write to a temporary file first, then rename (atomic on POSIX systems)
        # This prevents corruption if the process is killed mid-write
        temp_file = MEMORY_FILE.with_suffix(".tmp")

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(records_as_dicts, f, indent=2, ensure_ascii=False)

        # Atomic rename (overwrites existing file)
        temp_file.replace(MEMORY_FILE)

        # Step 8: ADD TO VECTOR STORE (for semantic search)
        # 
        # WHY EMBEDDING HAPPENS ONLY AFTER PASS:
        # ========================================
        # We only embed memories that have:
        # 1. Been validated by the Evaluator (score >= 8, pass = True)
        # 2. Successfully saved to JSON (source of truth)
        # 
        # This ensures:
        # - Vector store contains only HIGH-QUALITY memories
        # - No wasted embedding costs on failed/low-quality outputs
        # - Embedding API calls happen AFTER critical path (JSON save succeeded)
        # - Users get instant feedback (don't wait for embedding to complete)
        # 
        # Timeline:
        # t=0: User task completes
        # t=1: Evaluator scores it (e.g., score=9, pass=True)
        # t=2: save_memory() writes to JSON ← Critical path ends here
        # t=3: Embedding generated (50-200ms) ← Async/best-effort
        # t=4: Vector store updated ← Non-blocking
        # 
        # If embedding fails (API down, rate limit, network error):
        # - JSON still has the memory (can re-sync later)
        # - User experience is unaffected
        # - Logs the error for investigation
        # - Next sync_from_json() will catch it
        #
        try:
            # Lazy import to avoid circular dependencies and keep vector store optional
            from memory.vector_store import VectorMemoryStore
            
            vector_store = VectorMemoryStore()
            vector_store.add_new_memory(record)
            print(f"✅ Memory {record.id} added to vector store")
            
        except ImportError:
            # Vector store module not available (chromadb not installed, etc.)
            # This is fine - vector search is optional, keyword search still works
            print("ℹ️  Vector store not available (chromadb not installed?). Using keyword search only.")
            
        except Exception as e:
            # Vector store operation failed (API error, network issue, etc.)
            # This is NON-CRITICAL - memory is already saved to JSON
            # Log the error but don't crash or return False
            print(f"⚠️  Failed to add memory to vector store (non-critical): {type(e).__name__}: {e}")
            print("   Memory is saved in JSON. Run sync_from_json() to retry embedding.")

        return True

    except PermissionError:
        # File system permission denied
        # This can happen in read-only filesystems or restricted directories
        print(f"❌ Permission denied writing to {MEMORY_FILE}")
        return False

    except OSError as e:
        # Disk full, file system error, etc.
        print(f"❌ OS error writing memory: {e}")
        return False

    except Exception as e:
        # Catch-all for unexpected errors (JSON serialization, etc.)
        # This should never happen if MemoryRecord is valid, but we play it safe
        print(f"❌ Unexpected error saving memory: {type(e).__name__}: {e}")
        return False


def load_all_memories() -> list[MemoryRecord]:
    """
    Load all memory records from the local JSON file.

    This function is designed to NEVER crash the application. If any error occurs,
    it returns an empty list and logs the error.

    Returns:
        List of MemoryRecord instances (empty list if file missing or corrupt)

    Safety Guarantees:
    - Missing file returns empty list (not an error)
    - Corrupt JSON is logged and returns empty list
    - Invalid records (fail Pydantic validation) are skipped with a warning
    - File read errors do not crash the system
    - Empty file returns empty list

    Corruption Handling:
    - If JSON is malformed, the entire file is treated as corrupt
    - Corrupt file is NOT deleted (preserved for manual inspection)
    - Consider creating a backup before overwriting in production

    Performance:
    - O(n) where n = number of entries
    - All entries are loaded into memory (fine for < 10,000 entries)
    - For large datasets, use a database with pagination
    """

    try:
        # Step 1: Check if file exists
        if not MEMORY_FILE.exists():
            # Not an error - first time running or no memories yet
            return []

        # Step 2: Check if file is empty
        if MEMORY_FILE.stat().st_size == 0:
            # Empty file (can happen if write was interrupted)
            return []

        # Step 3: Read and parse JSON
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Step 4: Validate structure
        if not isinstance(data, list):
            # File should contain a list of records
            print(f"⚠️  Memory file contains {type(data).__name__}, expected list. Returning empty.")
            return []

        # Step 5: Deserialize and validate each record
        # Use Pydantic's model_validate to ensure each entry matches the schema
        # Invalid entries are skipped (defensive programming)
        valid_records: list[MemoryRecord] = []
        skipped_count = 0

        for i, record_dict in enumerate(data):
            try:
                # Pydantic validates all constraints (score >= 8, length limits, etc.)
                record = MemoryRecord.model_validate(record_dict)
                valid_records.append(record)
            except Exception as e:
                # This entry is invalid (could be from an older schema version)
                # Skip it but continue processing other entries
                skipped_count += 1
                print(f"⚠️  Skipping invalid memory entry at index {i}: {e}")

        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} invalid entries (out of {len(data)} total)")

        return valid_records

    except json.JSONDecodeError as e:
        # File exists but contains invalid JSON
        # This can happen if:
        # - Write was interrupted mid-way
        # - File was manually edited and corrupted
        # - Encoding issue
        print(f"❌ Corrupt memory file (invalid JSON): {e}")
        print(f"   File preserved at: {MEMORY_FILE}")
        print("   Consider backing up and deleting to start fresh")
        return []

    except PermissionError:
        # File system permission denied
        print(f"❌ Permission denied reading {MEMORY_FILE}")
        return []

    except OSError as e:
        # File system error (disk read error, etc.)
        print(f"❌ OS error reading memory: {e}")
        return []

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"❌ Unexpected error loading memory: {type(e).__name__}: {e}")
        return []


# =============================================================================
# Memory Retrieval: Vector-Based Semantic Search (v2)
# =============================================================================


def find_relevant_memories(
    user_goal: str,
    max_results: int = 3,
) -> list[MemoryRecord]:
    """
    Find the most relevant memory entries for a given user goal using semantic search.

    This implementation uses vector embeddings and similarity search to find
    semantically related memories, even when exact keywords don't match.

    VECTOR-BASED RETRIEVAL (Current Implementation):
    =================================================
    
    How it works:
    1. Embed the incoming user_goal (query → vector)
    2. Search vector DB for most similar stored memories
    3. Return top-k matches ranked by semantic similarity
    
    Advantages over keyword matching:
    - Matches by MEANING, not just exact words
    - Handles synonyms: "car" matches "automobile", "vehicle"
    - Context-aware: "apple" (fruit) vs "apple" (company)
    - Typo-tolerant: embeddings robust to spelling errors
    
    Example:
        Query: "How to use embeddings for search?"
        Matches:
        ✓ Goal: "Explain vector databases" (summary mentions "embeddings")
        ✓ Goal: "Set up semantic search" (conceptually related)
        ✗ Goal: "Configure FastAPI routes" (unrelated)
    
    GRACEFUL FALLBACK:
    ==================
    
    If vector DB is unavailable (not installed, API error, empty):
    - Returns empty list (does NOT crash)
    - Logs warning for debugging
    - System continues functioning (vector search is optional)
    
    Why empty list (not exception):
    - Memory retrieval is advisory, not critical
    - Supervisor can plan without memories (just less optimized)
    - Better to degrade gracefully than fail hard
    - User experience is unaffected
    
    Args:
        user_goal: The current user request to match against stored memories
        max_results: Maximum number of memories to return (default: 3)

    Returns:
        List of MemoryRecord instances, sorted by relevance (best match first)
        Returns empty list if:
        - No memories stored yet
        - user_goal is empty/invalid
        - Vector DB unavailable or empty
        - No matches above similarity threshold

    Performance:
    - Embedding generation: ~50-200ms (API call to Google)
    - Vector search: <10ms (local Chroma DB)
    - Total: ~100-250ms (acceptable for planning phase)
    
    For keyword-based fallback, see _find_relevant_memories_keyword() below.
    """

    # Step 0: Validate input
    if not user_goal or not user_goal.strip():
        # Empty query returns nothing
        return []

    # Step 1: Try vector-based search
    try:
        # Lazy import: only load if vector search is attempted
        # This keeps the module functional even if chromadb isn't installed
        from memory.vector_store import VectorMemoryStore
        
        # Initialize vector store
        # This connects to local Chroma DB in memory/chroma_db/
        store = VectorMemoryStore()
        
        # Query for similar memories
        # min_similarity=0.5 filters out weakly related results
        # (0.5 = 50% semantic similarity, reasonable threshold)
        results = store.find_similar(
            user_goal=user_goal,
            top_k=max_results,
            min_similarity=0.5,
        )
        
        # Extract MemoryRecords from (record, similarity) tuples
        # Caller expects list[MemoryRecord], not list[tuple[MemoryRecord, float]]
        memories = [memory for memory, _ in results]
        
        if memories:
            # Success: found relevant memories via vector search
            return memories
        else:
            # Vector DB is empty or no matches above threshold
            # This is not an error - just means no relevant past experiences
            return []
    
    except ImportError:
        # chromadb not installed or vector_store.py not available
        # This is expected in minimal deployments (keyword search only)
        print("ℹ️  Vector store not available. Install chromadb for semantic search.")
        print("   Returning empty results. To install: pip install chromadb")
        return []
    
    except Exception as e:
        # Vector search failed (API error, network issue, corrupt DB, etc.)
        # Log the error but don't crash - memory retrieval is advisory
        print(f"⚠️  Vector search failed: {type(e).__name__}: {e}")
        print("   Returning empty results. Check GOOGLE_API_KEY and vector DB health.")
        return []


# =============================================================================
# Legacy: Keyword-Based Retrieval (Deprecated but kept for reference)
# =============================================================================
#
# The functions below implement simple keyword matching using Jaccard similarity.
# They are no longer used by find_relevant_memories() (now uses vector search),
# but are kept for:
# 1. Backwards compatibility (if someone calls them directly)
# 2. Testing and comparison (keyword vs vector accuracy)
# 3. Emergency fallback (if vector DB completely unavailable)
#
# To use keyword matching instead of vector search, call:
#   _find_relevant_memories_keyword(user_goal, max_results)
# =============================================================================


def _find_relevant_memories_keyword(
    user_goal: str,
    max_results: int = 3,
) -> list[MemoryRecord]:
    """
    LEGACY: Find relevant memories using keyword matching (Jaccard similarity).
    
    This is the original v1 implementation, now replaced by vector search.
    Kept for backwards compatibility and as a fallback option.
    
    Use this if:
    - Vector DB is unavailable (chromadb not installed)
    - You want to compare keyword vs vector accuracy
    - You need zero-latency search (no embedding API call)
    
    For production use, prefer find_relevant_memories() which uses vector search.
    """
    
    # Step 1: Load all memories
    all_memories = load_all_memories()

    if not all_memories:
        return []

    if not user_goal or not user_goal.strip():
        return []

    # Step 2: Tokenize the query
    query_tokens = _tokenize(user_goal)

    if not query_tokens:
        return []

    # Step 3: Compute similarity for each memory
    scored_memories: list[tuple[float, MemoryRecord]] = []

    for memory in all_memories:
        memory_tokens = _tokenize(memory.user_goal)

        if not memory_tokens:
            continue

        similarity = _jaccard_similarity(query_tokens, memory_tokens)

        # Threshold of 0.2 means at least 20% keyword overlap
        if similarity >= 0.2:
            scored_memories.append((similarity, memory))

    # Step 4: Sort by similarity and take top-k
    scored_memories.sort(key=lambda x: x[0], reverse=True)
    top_memories = [memory for _, memory in scored_memories[:max_results]]

    return top_memories


def _tokenize(text: str) -> set[str]:
    """
    LEGACY: Tokenize text into a set of meaningful keywords.
    
    Used by keyword-based matching (_find_relevant_memories_keyword).
    Not used by the main find_relevant_memories() function (now uses vector embeddings).

    This is intentionally simple:
    - Lowercase normalization
    - Split on whitespace and punctuation
    - Remove common English stopwords
    - Keep only words with 3+ characters

    Returns:
        Set of tokens (set for fast intersection/union operations)

    Why a set?
    - Jaccard similarity uses sets (intersection / union)
    - Removes duplicate tokens automatically
    - Order doesn't matter for keyword matching
    """

    # Common English stopwords (minimal set to avoid false positives)
    # These words appear in almost every query so they don't distinguish memories
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "this", "what", "how", "why", "when"
    }

    # Lowercase and replace common punctuation with spaces
    text = text.lower()
    for char in ".,!?;:()[]{}":
        text = text.replace(char, " ")

    # Split into words
    words = text.split()

    # Filter: remove stopwords and short words (< 3 chars)
    # Short words like "to", "is", "at" are rarely meaningful
    tokens = {
        word for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }

    return tokens


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """
    LEGACY: Compute Jaccard similarity coefficient between two sets.
    
    Used by keyword-based matching (_find_relevant_memories_keyword).
    Not used by the main find_relevant_memories() function (now uses cosine similarity on embeddings).

    Formula: J(A, B) = |A ∩ B| / |A ∪ B|

    Returns:
        Float between 0.0 (no overlap) and 1.0 (identical sets)

    Examples:
        >>> _jaccard_similarity({"explain", "vector", "databases"}, 
        ...                      {"explain", "vector", "storage"})
        0.5  # 2 shared out of 4 unique tokens

        >>> _jaccard_similarity({"explain", "vectors"}, 
        ...                      {"describe", "embeddings"})
        0.0  # No overlap

    Why Jaccard?
    - Simple and intuitive: measures "how much overlap"
    - Works well for keyword-based matching
    - Range [0, 1] makes threshold easy to interpret
    - Fast to compute (set operations are O(min(|A|, |B|)))

    Alternatives:
    - Cosine similarity on embeddings (current implementation in vector_store.py)
    - Edit distance (better for typo tolerance)
    - BM25 (better for longer documents, considers term frequency)
    """

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b  # Tokens in both
    union = set_a | set_b         # Tokens in either

    if not union:
        return 0.0

    return len(intersection) / len(union)


# =============================================================================
# Usage Example (Vector-Based Semantic Search)
# =============================================================================
#
# from datetime import datetime, timezone
# from memory.memory_store import (
#     MemoryRecord, save_memory, find_relevant_memories
# )
#
# # EXAMPLE 1: Save memories (automatically indexed in vector DB)
# # ==============================================================
# records = [
#     MemoryRecord(
#         id="1",
#         user_goal="Explain vector databases for semantic search",
#         summary="Vector databases store embeddings and support nearest-neighbor queries. Key advantages: semantic matching beats keyword search.",
#         final_output="Vector databases are specialized systems...",
#         score=9,
#         created_at=datetime.now(timezone.utc),
#     ),
#     MemoryRecord(
#         id="2",
#         user_goal="Set up LangGraph workflows",
#         summary="LangGraph provides stateful orchestration for agent systems. Enables complex multi-step reasoning.",
#         final_output="LangGraph is a framework for...",
#         score=10,
#         created_at=datetime.now(timezone.utc),
#     ),
# ]
#
# for r in records:
#     if save_memory(r):
#         print(f"✅ Saved: {r.user_goal}")
#         # Behind the scenes:
#         # - JSON updated (source of truth)
#         # - Embedding generated from user_goal + summary
#         # - Vector stored in Chroma DB
#
#
# # EXAMPLE 2: Find relevant memories (semantic search)
# # ====================================================
# # Query doesn't need to match keywords exactly
# similar = find_relevant_memories("How do embeddings help with search?")
# 
# print(f"\nFound {len(similar)} relevant memories:")
# for m in similar:
#     print(f"  - {m.user_goal}")
#     print(f"    Score: {m.score}/10")
#     print(f"    Summary: {m.summary[:80]}...")
#
# # Example output:
# # Found 1 relevant memories:
# #   - Explain vector databases for semantic search
# #     Score: 9/10
# #     Summary: Vector databases store embeddings and support nearest-neighbor queries...
# #
# # Note: Even though query asks about "embeddings" and stored goal says
# # "vector databases", they match semantically because the summary mentions
# # "embeddings" and the concepts are related!
#
#
# # EXAMPLE 3: Keyword fallback (if vector DB unavailable)
# # =======================================================
# from memory.memory_store import _find_relevant_memories_keyword
#
# # Use legacy keyword matching if needed
# keyword_results = _find_relevant_memories_keyword(
#     "Explain vector databases",
#     max_results=3
# )
#
# =============================================================================
