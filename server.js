const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3862;

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', name: 'MiniLLM', version: '1.0.0' });
});

app.listen(PORT, () => {
  console.log(`MiniLLM running on port ${PORT}`);
});
