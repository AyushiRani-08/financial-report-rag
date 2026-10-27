# Diagnostic Error Signatures & Fixes

| Error Signature | Root Cause | Solution |
| :--- | :--- | :--- |
| `different vector dimensions 1536 and 384` | Model generates 384-dim, but pgvector table defined as `vector(1536)`. | Align `EMBEDDING_MODEL` or alter DB table column to match current model. |
| `could not obtain connection from pool` | Connection leak in `vector_store.py` or `xbrl_retriever.py`. | Verify `self._pool.putconn(conn)` runs inside a `finally` block on every query. |
| `KeyError in canonical_map` | Company 10-K uses non-standard GAAP taxonomy. | Add alias mapping to `utils/canonical_map.py` (e.g., map `Revenues` to `total_revenue`). |
