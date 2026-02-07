# OpenAlex API Debug Guide

## How to Check If API Calls Are Working

### Step 1: Check Server Logs

When you try to build a corpus, watch the server terminal. You should see:

```
INFO: Searching OpenAlex for: 'vision transformer' (limit: 25)
INFO: API URL: https://api.openalex.org/works?search=vision transformer&per_page=25
INFO: OpenAlex request successful: 5 results returned, 12345 total available
INFO: Sample result: An Image is Worth 16x16 Words: Transformers... (OpenAlex ID: W1234567890)
INFO: Successfully parsed 5 papers from 5 results
```

### Step 2: Test API Directly

Test if OpenAlex API is working:

```bash
# Test with curl
curl "https://api.openalex.org/works?search=vision%20transformer&per_page=5&mailto=test@example.com"
```

Or use Python:
```python
python test_openalex_api.py
```

### Step 3: Check What's Happening

**If you see "No results found":**

1. **Check server logs** for:
   - `"OpenAlex API response: 0 results returned"`
   - `"No results found for query"`
   - `"API returned 0 total matches"`

2. **Common causes:**
   - Query too specific
   - Rate limit hit (wait 5-10 minutes)
   - Network issue

3. **Solutions:**
   - Try simpler query: "vision transformer"
   - Wait 5-10 minutes if rate limited
   - Check internet connection

---

## What the Code Does Now

### Multi-Strategy Search:

1. **First try:** All keywords together
   - Query: "vision transformer medical imaging"
   
2. **If no results:** Try first 2 keywords
   - Query: "vision transformer"
   
3. **If still no results:** Try first keyword only
   - Query: "vision"

### Better Logging:

- Shows exact API URL being called
- Shows number of results returned
- Shows sample result title
- Shows parsing success/failure

### Error Handling:

- Detects rate limits (429)
- Auto-retries with backoff
- Handles malformed data
- Returns empty list instead of crashing

---

## Quick Test Queries

Try these in the corpus builder:

**✅ Should work:**
- "vision transformer"
- "medical imaging"
- "deep learning"
- "neural networks"

**❌ Might not work:**
- Full paper titles
- Very specific phrases
- Too many words

---

## Debugging Steps

1. **Check server logs** - Look for OpenAlex messages
2. **Try simple query** - "vision transformer"
3. **Test API directly** - Use curl or test script
4. **Check rate limit** - Wait 5-10 minutes
5. **Verify network** - Check internet connection

---

## Expected Log Output (Success)

```
INFO: Expanding from topic: 'vision transformer' -> keywords: ['vision transformer']
INFO: Searching OpenAlex for: 'vision transformer' (limit: 25)
INFO: API URL: https://api.openalex.org/works?search=vision transformer&per_page=25
INFO: OpenAlex request successful: 25 results returned, 50000 total available
INFO: Sample result: An Image is Worth 16x16 Words... (OpenAlex ID: W1234567890)
INFO: Successfully parsed 25 papers from 25 results
```

---

## Expected Log Output (Failure)

```
WARNING: No results found for query: 'vision transformer'
WARNING: API returned 0 total matches for this query
WARNING: Suggestions:
WARNING:   1. Try broader terms
WARNING:   2. Check if rate limited
WARNING:   3. Verify network connection
```

---

## If Still Not Working

1. **Check server terminal** for detailed logs
2. **Run test script**: `python test_openalex_api.py`
3. **Test API directly** with curl
4. **Wait 10 minutes** if rate limited
5. **Try different query** - simpler terms

The code is now much more robust and will tell you exactly what's happening! 🚀

