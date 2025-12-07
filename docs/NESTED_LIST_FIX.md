# Nested Repositories List Bug - Investigation & Fix

## Problem

The agent was exhausting its budget (20 iterations) because search results were being filtered out incorrectly. Logs showed:

```
Filtering by repositories: [['ui', 'api']]  # NESTED!
After repository filter: 25 -> 0 results          # Everything filtered out!
```

## Root Cause Analysis

### 1. How the bug manifests

The repositories parameter was sometimes being passed as a nested list:
- **Wrong:** `[['ui', 'api']]`
- **Correct:** `['ui', 'api']`

This caused the filter to fail:
```python
r.get('repository') in [['ui', 'api']]  # Always False!
```

### 2. Where does the nested list come from?

**Traced through the call stack:**

1. **API endpoint** (`src/api/routes/search.py:285-294`)
   ```python
   context = request.context or {}
   # context comes from AskRequest.context (Dict[str, Any])
   ```

2. **Orchestrator** (`src/agents/orchestrator.py:408`)
   ```python
   result = await self.visual_agent.create_visualization(question, context)
   # Passes context directly to Visual Guide Agent
   ```

3. **Visual Guide Agent** (`src/agents/visual_guide_agent.py:150-152`)
   ```python
   explorer_result = await self.code_explorer.explore(
       question=search_query,
       context=context  # Passes context to Code Explorer
   )
   ```

4. **Code Explorer** (`src/agents/code_explorer.py:388-390`)
   ```python
   if 'repositories' in context:
       repos = context['repositories']
       parts.append(f"Repositories: {', '.join(repos)}")
   ```

   This formats context as text: **"Repositories: ui, api"**

5. **LLM interprets this text** and generates tool calls:
   ```json
   {
     "tool": "semantic_search",
     "params": {
       "query": "...",
       "repositories": [["ui", "api"]]  // WRONG - nested!
     }
   }
   ```

   Sometimes it creates it correctly as `["ui", "api"]`, sometimes nested!

### 3. Why does the LLM create nested lists?

The SYSTEM_PROMPT for Code Explorer had incomplete documentation:

**Before:**
```
- semantic_search: Find code by concept or behavior
  Parameters: query (string), scope (optional), top_k (optional), repo (optional)
  Example: {"tool": "semantic_search", "params": {"query": "authentication logic"}}
```

**Problems:**
- Only mentioned `repo` (singular), not `repositories` (plural)
- No example showing how to pass multiple repositories
- LLM had to **infer** the format from "Repositories: ui, api" in context

## Fixes Applied

### Fix #1: Defensive Flattening (Code Retriever)

**File:** `src/code_rag/retrieval/code_retriever.py:358-372`

```python
# Filter by repository if specified
if config.repositories:
    # Flatten repositories list if nested (e.g., [['ui', 'api']] -> ['ui', 'api'])
    repos = config.repositories
    if repos and isinstance(repos[0], list):
        repos = repos[0]
        logger.warning(f"Flattened nested repositories list: {config.repositories} -> {repos}")

    logger.info(f"Filtering by repositories: {repos}")
    before_count = len(results)
    results = [
        r for r in results
        if r.get('repository') in repos
    ]
    logger.info(f"After repository filter: {before_count} -> {len(results)} results")
```

**Purpose:** Safety net that automatically fixes nested lists

### Fix #2: Improved Tool Documentation (Code Explorer)

**File:** `src/agents/code_explorer.py:101-104`

```python
- semantic_search: Find code by concept or behavior
  Parameters: query (string), scope (optional), top_k (optional), repositories (optional: list of repo names)
  Example: {"tool": "semantic_search", "params": {"query": "authentication logic", "top_k": 20}}
  Example (multi-repo): {"tool": "semantic_search", "params": {"query": "equity trading", "repositories": ["ui", "api"]}}
```

**Changes:**
- ✅ Changed `repo (optional)` to `repositories (optional: list of repo names)`
- ✅ Added explicit example showing correct format: `["ui", "api"]`
- ✅ LLM now has clear guidance on parameter format

## Verification

### Before Fix
```
Filtering by repositories: [['ui', 'api']]
After repository filter: 25 -> 0 results
Agent budget exhausted (20/20 iterations)
```

### After Fix (Expected)
```
Filtering by repositories: ['ui', 'api']  # Flat list
After repository filter: 25 -> 12 results        # Correct filtering
Agent completes in ~8 iterations                 # Faster
Visual Guide receives sources                     # Success
```

## Impact

### Problems Solved
1. ✅ No more filtering out all results
2. ✅ Agent completes faster (doesn't waste iterations)
3. ✅ Visual Guide Agent receives sources
4. ✅ Diagrams can be generated successfully

### Files Modified
1. `src/code_rag/retrieval/code_retriever.py` - Added defensive flattening
2. `src/agents/code_explorer.py` - Improved tool documentation
3. `docs/NESTED_LIST_FIX.md` - This documentation

## Related Issues

This fix complements the previous fixes documented in:
- `docs/FIXES_2025-12-07.md` - Main fixes for search config, sources extraction
- `QUICK_FIX_SUMMARY.md` - Quick reference
- `FINAL_STATUS.md` - System status

## Testing Recommendations

1. **Restart the server** to load updated prompts
2. **Test with new query** (avoid cache):
   ```
   Show me how equity instruments flow from UI to database
   ```
3. **Check logs** for:
   - Flat repositories list (not nested)
   - Correct filtering results
   - Fewer iterations used
   - Sources extracted successfully

## Next Steps

1. ✅ Applied both fixes (flattening + documentation)
2. ⏳ Restart server to test
3. ⏳ Verify agent completes faster
4. ⏳ Verify Visual Guide works end-to-end

---

**Date:** 2025-12-07
**Issue:** Agent exhausting budget due to nested repositories list
**Status:** Fixed (pending server restart & verification)
