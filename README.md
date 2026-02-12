# MiniLLM

An interactive platform for learning how neural networks and transformers work — from the ground up. Train character-level language models in your browser and watch them learn in real time.

**Live:** [minillm.llm.kaveenk.com](https://minillm.llm.kaveenk.com)

## Features

- 🧠 **Interactive Neural Network Visualizer** — Watch neurons fire, weights update, and gradients flow
- 📝 **Character-Level Language Models** — Train RNNs on Shakespeare, recipes, Python code
- 🎮 **Live Training** — Real-time loss curves, generated text samples at each epoch
- 🔤 **Tokenization Explorer** — Type a sentence, see it tokenized and embedded into vectors
- 🏗️ **Custom Model Training** — Upload your own text, train a model, generate text
- 🎓 **Educational Explanations** — Step-by-step breakdowns of backpropagation, attention, embeddings
- 🔒 **Turnstile Protection** — Rate-limited API with Cloudflare Turnstile verification

## Tech Stack

- **Runtime:** Node.js
- **Framework:** Express 5
- **Database:** SQLite (better-sqlite3) for custom model storage
- **Frontend:** Vanilla HTML/CSS/JS with Canvas visualizations
- **ML:** Custom character-level RNN implementation in JavaScript
- **Auth:** Cloudflare Turnstile

## Setup

```bash
npm install
cp .env.example .env
node server.js
```

### .env.example

```env
PORT=3860
TURNSTILE_SECRET=your-cloudflare-turnstile-secret
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available pre-trained models |
| POST | `/api/generate` | Generate text from a model |
| POST | `/api/train` | Train a custom model on user text |
| GET | `/api/health` | Health check |

## Pre-trained Models

- **Shakespeare** — Trained on Shakespeare's works
- **Recipes** — Trained on cooking recipes
- **Python** — Trained on Python source code

## License

MIT
