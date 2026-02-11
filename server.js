const express = require('express');
const path = require('path');
const crypto = require('crypto');
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

// Rate limiting (in-memory)
const rateLimits = new Map();
function checkRateLimit(ip, maxReqs, windowMs) {
  const now = Date.now();
  const key = `${ip}`;
  const entry = rateLimits.get(key) || { count: 0, resetAt: now + windowMs };
  if (now > entry.resetAt) { entry.count = 0; entry.resetAt = now + windowMs; }
  entry.count++;
  rateLimits.set(key, entry);
  return entry.count <= maxReqs;
}

// Middleware
app.use(express.json({ limit: '6mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Health
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', name: 'MiniLLM', version: '2.0.0' });
});

// Save model
app.post('/api/models/save', (req, res) => {
  const ip = req.headers['x-forwarded-for'] || req.ip;

  // Rate limit: 5 saves per hour
  if (!checkRateLimit(ip, 5, 3600000)) {
    return res.status(429).json({ error: 'Rate limit exceeded. Max 5 saves per hour.' });
  }

  const data = JSON.stringify(req.body);
  const sizeBytes = Buffer.byteLength(data);

  // Max 5MB
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

// Model info (without full weights)
app.get('/api/models/:id/info', (req, res) => {
  const row = db.prepare('SELECT preset, created_at, size_bytes FROM models WHERE id = ?').get(req.params.id);
  if (!row) return res.status(404).json({ error: 'Model not found.' });
  res.json({ id: req.params.id, preset: row.preset, createdAt: row.created_at, sizeBytes: row.size_bytes });
});

app.listen(PORT, () => {
  console.log(`MiniLLM running on port ${PORT}`);
});
