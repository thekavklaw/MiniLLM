# MiniLLM — Learn How AI Thinks

## Vision
An interactive learning platform that takes someone who only knows "ChatGPT is an AI" and teaches them end-to-end how neural networks and LLMs work — through beautiful, hands-on visualizations.

NOT a terminal/hacker theme. Think: soft gradients, floating particles, cloud-like backgrounds, warm colors, approachable typography. Like a modern textbook meets an art installation.

## Competitive Landscape
- **TensorFlow Playground**: The gold standard, but intimidating. No explanations, just knobs. Assumes ML knowledge.
- **ConvNetJS**: Karpathy's demo. Powerful but ugly, no guidance.
- **Transformer Explainer (Georgia Tech)**: Great concept but complex, assumes CS background.
- **Jay Alammar's Illustrated Transformer**: Beautiful static diagrams but not interactive.

## Our Edge
1. **Zero-knowledge starting point** — assumes you've only used ChatGPT
2. **Guided journey** — not just a sandbox, but a story with chapters
3. **Beautiful design** — airy, modern, unique (not another dark terminal UI)
4. **Build AND understand** — you construct networks yourself and watch them learn
5. **All the way to transformers** — from a single neuron to attention mechanisms

## Architecture

### Frontend (vanilla JS + Canvas/WebGL)
- **Chapter-based learning path** with interactive visualizations
- **Neural network canvas** — drag neurons, connect layers, watch training
- **Real-time gradient descent visualization**
- **Attention mechanism explorer**
- **Token/embedding visualizer**

### Backend (Node.js + Express)
- Serve static frontend
- API for heavier computations (optional, for transformer demos)
- Integration with a small model for live inference demos

## Chapters (Learning Journey)

### Chapter 1: What IS a Neural Network?
- Start with: "You've used ChatGPT. Here's what's actually happening inside."
- Interactive: Single neuron — drag a slider for input, watch it compute output
- Explain: weights, bias, activation function with VISUAL metaphors (not math)
- Activity: Build your first neuron that detects "is this number > 5?"

### Chapter 2: Neurons That Learn
- "How does a neuron learn the right answer?"
- Interactive: Watch a neuron adjust its weights to fit data points
- Gradient descent as "rolling a ball downhill" — animated loss landscape
- Activity: Train a single neuron on AND gate (2 inputs → 1 output)

### Chapter 3: Layers of Thinking
- "One neuron isn't smart enough. What if we stack them?"
- Interactive: Build a 2-layer network by dragging neurons onto a canvas
- Watch XOR problem: single neuron fails, 2-layer solves it
- Real-time visualization: see the decision boundary morph during training

### Chapter 4: The Playground (TF Playground-style but better)
- Full sandbox: add/remove layers, neurons, change activation functions
- Datasets: Circle, XOR, Spiral, Gaussian, Checkerboard, Moon
- Real-time training with decision boundary visualization
- Presets for common architectures
- "Explain what's happening" panel that narrates in plain English

### Chapter 5: How Computers Read Words
- "ChatGPT doesn't see words — it sees numbers"
- Interactive: Type a sentence, watch it tokenize, then embed into vectors
- 2D/3D embedding visualization — drag words around, see which are "close"
- Activity: Find words that are similar in embedding space

### Chapter 6: Paying Attention
- "The key insight behind ChatGPT: attention"
- Interactive attention visualization — type a sentence, see which words attend to which
- Query/Key/Value explained with a library metaphor (searching for books)
- Self-attention heatmap you can interact with

### Chapter 7: The Transformer
- Put it all together: embeddings → attention → feed-forward → output
- Step through a tiny transformer processing "The cat sat on the ___"
- Watch each layer transform the representation
- See how it predicts the next word

### Chapter 8: From Tiny to GPT
- Scale visualization: our tiny network vs GPT-4 (parameter count comparison)
- How training data shapes behavior
- RLHF explained visually
- Interactive: Chat with a tiny model and see its "thought process"

## Design Language
- **Background**: Soft gradient (lavender → sky blue → white), floating particle clouds
- **Typography**: Modern sans-serif (Inter), generous whitespace, large readable text
- **Colors**: Soft purples, blues, warm oranges for active elements, white cards
- **Animations**: Smooth, gentle, physics-based (spring animations)
- **Cards**: Glassmorphic with subtle blur
- **Neurons**: Glowing orbs with soft shadows, connected by animated flowing lines
- **No jargon without explanation** — every technical term has a tooltip or inline explanation

## Tech Stack
- Node.js + Express backend (port 3862)
- Vanilla JS + Canvas for neural network visualization
- CSS animations for UI
- Optional: TensorFlow.js for browser-side training
- Optional: ONNX.js for running a tiny transformer model

## Deployment
- Port: 3862
- systemd: `minillm.service`
- nginx: `mini.llm.kaveenk.com`
- Cloudflare: CNAME + Advanced Certificate
- GitHub: `Kav-K/MiniLLM` (PUBLIC)
