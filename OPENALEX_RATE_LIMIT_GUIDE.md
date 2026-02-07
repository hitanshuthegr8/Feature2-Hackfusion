# OpenAlex API Rate Limit Guide

## What I Fixed

✅ **Rate limit detection** - Now detects 429 (Too Many Requests) errors
✅ **Automatic retry** - Waits and retries when rate limited
✅ **Better search strategy** - Uses space-separated queries instead of `|`
✅ **Fallback search** - Tries simpler query if first one fails
✅ **Better logging** - Shows exactly what's happening

---

## OpenAlex Rate Limits

**Free tier limits:**
- **100,000 requests per day** (per email)
- **10 requests per second** (burst)
- **No hard limit** but be respectful

**If you hit the limit:**
- You'll get HTTP 429 error
- Wait 1-2 minutes before trying again
- The code now automatically retries

---

## How to Avoid Rate Limits

### 1. Use Your Email (Recommended)

Set your email in environment variable:
```bash
# Windows PowerShell
$env:OPENALEX_EMAIL="your-email@example.com"

# Or add to .env file
OPENALEX_EMAIL=your-email@example.com
```

**Benefits:**
- Higher rate limits
- Better API performance
- Polite API usage

### 2. Wait Between Requests

The code now waits 0.2 seconds between requests (5 per second max).

### 3. Use Broader Topics

**❌ Too specific:**
- "Vision Transformer for Medical Image Segmentation using BraTS 2021 Dataset"

**✅ Better:**
- "vision transformer medical imaging"
- "medical image segmentation"
- "vision transformer"

**✅ Best:**
- "vision transformer"
- "medical imaging"
- "image segmentation"

---

## What to Do If You Hit Rate Limit

### Option 1: Wait and Retry (Automatic)

The code now automatically:
1. Detects 429 error
2. Reads `Retry-After` header
3. Waits the required time
4. Retries up to 3 times

**Just wait 1-2 minutes and try again!**

### Option 2: Check Your Usage

OpenAlex doesn't provide a usage dashboard, but you can:
1. Check server logs for 429 errors
2. Wait 5-10 minutes if you've made many requests
3. Try again with a simpler query

### Option 3: Use Different Email

If you have multiple emails, you can:
1. Set `OPENALEX_EMAIL` to a different email
2. Restart the server
3. Try again

---

## Testing If Rate Limit is the Issue

### Test 1: Simple Query

Try a very simple, common topic:
```
vision transformer
```

If this works, your rate limit is fine - the issue is query format.

### Test 2: Check Server Logs

Look for these messages:
- `"OpenAlex rate limit hit"` - You're rate limited
- `"OpenAlex request failed: 429"` - Rate limit confirmed
- `"No results found"` - Query issue, not rate limit

### Test 3: Direct API Test

Test OpenAlex directly:
```bash
curl "https://api.openalex.org/works?search=vision%20transformer&per_page=5&mailto=your-email@example.com"
```

If this returns results, API is working.

---

## Best Practices

### ✅ DO:
- Use 2-4 key terms
- Use general topics first
- Wait between large batch requests
- Set your email in environment variable
- Check server logs for errors

### ❌ DON'T:
- Use full paper titles
- Make rapid-fire requests
- Use very specific queries
- Ignore 429 errors

---

## Example Good Queries

**For medical imaging:**
- ✅ "medical image segmentation"
- ✅ "medical imaging deep learning"
- ✅ "brain tumor segmentation"

**For vision transformers:**
- ✅ "vision transformer"
- ✅ "transformer image classification"
- ✅ "ViT medical imaging"

**For general AI:**
- ✅ "deep learning"
- ✅ "neural networks"
- ✅ "machine learning"

---

## Troubleshooting

### "No papers found" for every query?

**Possible causes:**
1. **Rate limit hit** - Wait 5-10 minutes
2. **Query too specific** - Use broader terms
3. **Network issue** - Check internet connection
4. **API temporarily down** - Try again later

**Solution:**
1. Check server logs (look for 429 errors)
2. Try simplest query: "deep learning"
3. Wait 5 minutes
4. Try again

### Still not working?

1. **Check server logs** - Look for OpenAlex errors
2. **Test API directly** - Use curl command above
3. **Wait longer** - Sometimes need 10-15 minutes
4. **Use different email** - Set OPENALEX_EMAIL

---

## Quick Fix Checklist

- [ ] Wait 5-10 minutes
- [ ] Try simpler query: "vision transformer"
- [ ] Check server logs for 429 errors
- [ ] Set OPENALEX_EMAIL environment variable
- [ ] Restart server
- [ ] Try again

---

## Code Changes Made

1. **Rate limit detection** - Detects 429 status
2. **Automatic retry** - Waits and retries automatically
3. **Better search** - Uses space-separated queries
4. **Fallback strategy** - Tries simpler query if first fails
5. **Better logging** - Shows what's happening

The code is now much more resilient to rate limits! 🚀

