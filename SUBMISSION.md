# SUBMISSION RUNBOOK — do this tomorrow (≈20 min)

You need to submit **two things**: (1) a public API URL where `/health` and
`/chat` work, and (2) the approach document (`APPROACH.pdf`, already built).

Everything below is copy-paste. Do steps 1→4 in order.

---

## Step 1 — Get a free LLM key (2 min)
The service runs without one (heuristic fallback), but a real model makes the
clarify/compare/refine behavior and Recall much stronger. Use **Groq** (free, fast):

1. Go to <https://console.groq.com> → sign in → **API Keys** → **Create API Key**.
2. Copy the key (starts with `gsk_...`). You'll paste it in Step 3.

*(Alternatives if Groq is down: OpenRouter free tier, or Google AI Studio /
Gemini. Any OpenAI-compatible endpoint works — just change `LLM_BASE_URL` and
`LLM_MODEL`.)*

## Step 2 — Push to GitHub (5 min)
```bash
cd shl-recommender
git init && git add . && git commit -m "SHL conversational recommender"
# create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/shl-recommender.git
git branch -M main && git push -u origin main
```

## Step 3 — Deploy on Render (8 min, free, `render.yaml` already included)
1. Go to <https://render.com> → sign in with GitHub → **New → Web Service**.
2. Pick your `shl-recommender` repo. Render auto-detects `render.yaml`.
3. Under **Environment**, add one secret:
   - `LLM_API_KEY` = the `gsk_...` key from Step 1.
   *(`LLM_BASE_URL` and `LLM_MODEL` are already set by `render.yaml`.)*
4. Click **Create Web Service**. Wait for the build to go green.
5. Your URL is `https://<name>.onrender.com`. **Health check path is `/health`.**

> Free Render instances cold-start (~30–60s). The assignment explicitly allows up
> to 2 min for the first `/health`, so this is fine.

## Step 4 — Verify both endpoints (2 min)
Replace `<URL>` with your Render URL:
```bash
curl <URL>/health
# -> {"status":"ok"}

curl -X POST <URL>/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer who works with SQL"}]}'
# -> {"reply":"...", "recommendations":[{"name":...,"url":"https://www.shl.com/...","test_type":"K"}], "end_of_conversation":false}
```
If `/chat` returns recommendations with `shl.com` URLs, you're done.

## Step 5 — Submit
On the assignment form, provide:
- **API endpoint URL:** your Render base URL (they will call `/health` and `/chat`).
- **Approach document:** upload `APPROACH.pdf` (2 pages).

---

## Deploy alternatives (if Render gives trouble)
- **Railway:** New Project → Deploy from GitHub → add `LLM_API_KEY` var. Start
  command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **Hugging Face Spaces (Docker):** create a Docker Space, push this repo (it has a
  `Dockerfile`), add `LLM_API_KEY` as a Space secret.
- **Fly.io:** `fly launch` (uses the `Dockerfile`), `fly secrets set LLM_API_KEY=...`.

## One decision to be aware of
The traces render `recommendations: null` while clarifying, but the spec *text*
says "EMPTY". I return `[]` (empty array) — safer for a strict Pydantic evaluator
and consistent with the response example in the PDF. If you later learn their
grader wants literal `null`, it's a one-line change in `app/schemas.py`
(`recommendations: list[...] | None`) and `app/agent/orchestrator.py`.

## Sanity commands (optional, before pushing)
```bash
pip install -r requirements.txt
python tests/test_agent.py        # expect 11/11
python -m app.eval.replay         # expect mean recall printed (~0.60 offline; higher with a key)
```
