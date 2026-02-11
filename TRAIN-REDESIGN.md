# Train Your Own — Redesign

## Problem
- Character-level LSTM crashes browser, generates garbage
- Too compute-heavy for frontend
- Not interactive enough

## New Approach: Interactive Classifier Builder

### Concept
Instead of "train a language model" (boring, slow, garbage output), do:
**"Teach AI to understand YOUR categories"**

User creates categories, provides examples, trains instantly, tests live.

### User Flow
1. **Name your categories** (e.g., "Happy" and "Sad", or "Spam" and "Not Spam", or custom)
2. **Add examples** — type sentences and drag them to categories (or click to assign)
3. **Train** — instant (< 1 second), frontend TF.js, tiny model
4. **Test** — type new sentences, see classification in real-time with confidence bars
5. **Share** — unique URL

### Presets (one-click demos)
- 🎭 Sentiment: Happy vs Sad (pre-loaded with 20 examples each)
- 📧 Spam Filter: Spam vs Not Spam
- 🐱 Cat vs Dog: Descriptions of cats vs dogs
- 🌤️ Weather: Sunny vs Rainy descriptions

### Technical
- **Frontend**: TF.js with simple bag-of-words → 2-layer dense network
- Tokenize with simple word splitting + frequency vocabulary
- Model: Input(vocab_size) → Dense(32, relu) → Dense(num_classes, softmax)
- Training: 10-50 epochs, batch size = all data, < 1 second
- **No backend needed for training** — it's tiny

### ALSO: "AI Completes Your Sentence" Demo
- Backend endpoint: `/api/complete`
- Uses a TINY pre-trained model (or even just n-gram/markov chain)
- Stateless, fast, limited to 50 tokens
- Rate limited: 10 requests/min/IP
- This gives the "wow, AI writes!" moment without browser-crashing LSTM

### Backend (lightweight)
- Markov chain text generator (no GPU, pure JS, instant)
- Pre-built from Shakespeare/recipes/code at startup
- `/api/complete?text=...&preset=shakespeare` → returns continuation
- Capped at 100 chars output, 10 req/min
