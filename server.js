const express = require('express');
const path = require('path');
const crypto = require('crypto');
const fs = require('fs');
const Database = require('better-sqlite3');
const app = express();
const PORT = process.env.PORT || 3862;

// Database
const db = new Database(path.join(__dirname, 'minillm.db'));
db.pragma('journal_mode = WAL');
db.exec(`
  CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    preset TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    ip TEXT,
    size_bytes INTEGER
  )
`);

// Rate limiting
const rateLimits = new Map();
function checkRateLimit(ip, maxReqs, windowMs) {
  const now = Date.now();
  const entry = rateLimits.get(ip) || { count: 0, resetAt: now + windowMs };
  if (now > entry.resetAt) { entry.count = 0; entry.resetAt = now + windowMs; }
  entry.count++;
  rateLimits.set(ip, entry);
  return entry.count <= maxReqs;
}

// ===== Markov Chain (order-4 character level) =====
const MARKOV_ORDER = 4;
const markovChains = {};

function buildMarkovChain(text, order) {
  const chain = {};
  for (let i = 0; i <= text.length - order - 1; i++) {
    const key = text.substring(i, i + order);
    const next = text[i + order];
    if (!chain[key]) chain[key] = {};
    chain[key][next] = (chain[key][next] || 0) + 1;
  }
  return chain;
}

function generateFromChain(chain, seed, length, order) {
  let current = seed.slice(-order);
  if (!chain[current]) {
    for (let len = order - 1; len >= 1; len--) {
      const partial = seed.slice(-len);
      const match = Object.keys(chain).find(k => k.startsWith(partial));
      if (match) { current = match; break; }
    }
    if (!chain[current]) {
      const keys = Object.keys(chain);
      current = keys[Math.floor(Math.random() * keys.length)];
    }
  }
  let result = '';
  for (let i = 0; i < length; i++) {
    const options = chain[current];
    if (!options) break;
    const entries = Object.entries(options);
    const total = entries.reduce((s, [, c]) => s + c, 0);
    let r = Math.random() * total;
    let next = entries[0][0];
    for (const [char, count] of entries) {
      r -= count;
      if (r <= 0) { next = char; break; }
    }
    result += next;
    current = current.slice(1) + next;
  }
  return result;
}

// Build chains at startup
const dataDir = path.join(__dirname, 'public', 'data');
['shakespeare', 'recipes', 'python'].forEach(name => {
  try {
    const text = fs.readFileSync(path.join(dataDir, `${name}.txt`), 'utf-8');
    markovChains[name] = buildMarkovChain(text, MARKOV_ORDER);
    console.log(`Markov chain: ${name} (${Object.keys(markovChains[name]).length} states)`);
  } catch (e) {
    console.error(`Failed to build chain for ${name}:`, e.message);
  }
});

// Middleware
app.use(express.json({ limit: '6mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Health
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', name: 'MiniLLM', version: '3.0.0' });
});

// Markov completion
app.get('/api/complete', (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;
  if (!checkRateLimit(ip, 10, 60000)) {
    return res.status(429).json({ error: 'Rate limited. Max 10 requests per minute.' });
  }
  const { text, preset, length } = req.query;
  if (!text || !preset) return res.status(400).json({ error: 'Missing text or preset.' });
  const chain = markovChains[preset];
  if (!chain) return res.status(404).json({ error: `Unknown preset: ${preset}` });
  const maxLen = Math.min(parseInt(length) || 100, 200);
  res.json({ text: generateFromChain(chain, text, maxLen, MARKOV_ORDER) });
});

// Save model
app.post('/api/models/save', (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;
  if (!checkRateLimit(ip, 5, 3600000)) {
    return res.status(429).json({ error: 'Rate limit exceeded. Max 5 saves per hour.' });
  }
  const data = JSON.stringify(req.body);
  const sizeBytes = Buffer.byteLength(data);
  if (sizeBytes > 5 * 1024 * 1024) {
    return res.status(413).json({ error: 'Model too large. Max 5MB.' });
  }
  const id = crypto.randomBytes(8).toString('hex');
  try {
    db.prepare('INSERT INTO models (id, data, preset, ip, size_bytes) VALUES (?, ?, ?, ?, ?)')
      .run(id, data, req.body.preset || 'unknown', ip, sizeBytes);
    res.json({ id, url: `/model.html?id=${id}` });
  } catch (e) {
    console.error('Save error:', e);
    res.status(500).json({ error: 'Failed to save model.' });
  }
});

// Load model
app.get('/api/models/:id', (req, res) => {
  const row = db.prepare('SELECT data, preset, created_at FROM models WHERE id = ?').get(req.params.id);
  if (!row) return res.status(404).json({ error: 'Model not found.' });
  res.json({ ...JSON.parse(row.data), createdAt: row.created_at });
});

// Model info
app.get('/api/models/:id/info', (req, res) => {
  const row = db.prepare('SELECT preset, created_at, size_bytes FROM models WHERE id = ?').get(req.params.id);
  if (!row) return res.status(404).json({ error: 'Model not found.' });
  res.json({ id: req.params.id, preset: row.preset, createdAt: row.created_at, sizeBytes: row.size_bytes });
});

// ===== LSTM Neural Network Language Models =====
let tf;
const lstmModels = {};

async function loadLSTMModels() {
  try {
    tf = require('@tensorflow/tfjs-node');
  } catch (e) {
    console.error('TF.js not available, LSTM generation disabled');
    return;
  }
  const modelsDir = path.join(__dirname, 'models');
  for (const preset of ['shakespeare', 'recipes', 'python']) {
    const modelPath = path.join(modelsDir, preset, 'model.json');
    const vocabPath = path.join(modelsDir, preset, 'vocab.json');
    if (!fs.existsSync(modelPath) || !fs.existsSync(vocabPath)) {
      console.log(`LSTM model not found for ${preset}, skipping`);
      continue;
    }
    try {
      const model = await tf.loadLayersModel(`file://${modelPath}`);
      const vocab = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));
      lstmModels[preset] = { model, vocab };
      console.log(`LSTM model loaded: ${preset} (vocab=${vocab.vocabSize})`);
    } catch (e) {
      console.error(`Failed to load ${preset}:`, e.message);
    }
  }
}

function lstmGenerate(preset, seed, length, temperature) {
  const { model, vocab } = lstmModels[preset];
  const { charToIdx, idxToChar, vocabSize, seqLen } = vocab;
  temperature = temperature || 0.7;

  // Pad or truncate seed to seqLen
  let ctx = seed.slice(-seqLen);
  while (ctx.length < seqLen) ctx = ' ' + ctx;

  let result = '';
  for (let i = 0; i < length; i++) {
    // One-hot encode context
    const x = [];
    for (let j = 0; j < seqLen; j++) {
      const oh = new Float32Array(vocabSize);
      const idx = charToIdx[ctx[j]];
      if (idx !== undefined) oh[idx] = 1;
      else oh[0] = 1; // fallback
      x.push(oh);
    }

    const input = tf.tensor3d([x]);
    const pred = model.predict(input);
    const probs = pred.dataSync();
    input.dispose();
    pred.dispose();

    // Temperature sampling
    const scaled = Array.from(probs).map(p => Math.exp(Math.log(Math.max(p, 1e-8)) / temperature));
    const sum = scaled.reduce((a, b) => a + b, 0);
    const norm = scaled.map(s => s / sum);

    let r = Math.random();
    let idx = 0;
    for (let j = 0; j < norm.length; j++) {
      r -= norm[j];
      if (r <= 0) { idx = j; break; }
    }

    const ch = idxToChar[String(idx)] || ' ';
    result += ch;
    ctx = ctx.slice(1) + ch;
  }
  return result;
}

// LSTM generation endpoint
app.post('/api/generate', (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;
  if (!checkRateLimit(ip, 15, 60000)) {
    return res.status(429).json({ error: 'Rate limited. Max 15 requests per minute.' });
  }

  const { prompt, preset, temperature, length } = req.body;
  if (!prompt || !preset) return res.status(400).json({ error: 'Missing prompt or preset.' });
  if (!lstmModels[preset]) return res.status(404).json({ error: `Model "${preset}" not loaded.` });

  const maxLen = Math.min(parseInt(length) || 150, 300);
  const temp = Math.max(0.1, Math.min(2.0, parseFloat(temperature) || 0.7));

  try {
    const text = lstmGenerate(preset, prompt, maxLen, temp);
    res.json({ text, model: 'lstm-64', params: 35900, preset });
  } catch (e) {
    console.error('Generation error:', e);
    res.status(500).json({ error: 'Generation failed.' });
  }
});

// Model info endpoint
app.get('/api/model-info', (req, res) => {
  const loaded = Object.keys(lstmModels);
  res.json({
    models: loaded,
    architecture: 'LSTM (64 units)',
    totalParams: 35900,
    type: 'Character-level neural language model'
  });
});

// Load models then start server
loadLSTMModels().then(() => {
  app.listen(PORT, () => console.log(`MiniLLM running on port ${PORT}`));
}).catch(e => {
  console.error('Failed to load models:', e);
  app.listen(PORT, () => console.log(`MiniLLM running on port ${PORT} (no LSTM models)`));
});
