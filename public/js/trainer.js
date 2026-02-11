// MiniLLM — Character-level N-gram Language Model (pure JS, no TensorFlow)
(function() {
  'use strict';

  // ===== State =====
  let trainingData = '';
  let activePreset = 'shakespeare';
  let ngramModel = null; // Map: context string → { char: count, ... }
  let order = 5;
  let isGenerating = false;

  const presets = {
    shakespeare: '/data/shakespeare.txt',
    recipes: '/data/recipes.txt',
    python: '/data/python.txt'
  };

  // ===== Preset buttons =====
  document.querySelectorAll('#train-step-1 .preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#train-step-1 .preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activePreset = btn.dataset.preset;
      const customArea = document.getElementById('custom-text');
      if (activePreset === 'custom') {
        customArea.style.display = 'block';
        trainingData = customArea.value;
        updatePreview();
      } else {
        customArea.style.display = 'none';
        loadPreset(activePreset);
      }
      resetModel();
    });
  });

  const customArea = document.getElementById('custom-text');
  if (customArea) {
    customArea.addEventListener('input', () => {
      trainingData = customArea.value;
      updatePreview();
      resetModel();
    });
  }

  async function loadPreset(name) {
    if (!presets[name]) return;
    try {
      const resp = await fetch(presets[name]);
      trainingData = await resp.text();
      updatePreview();
    } catch (e) {
      console.error('Failed to load preset:', e);
    }
  }

  function updatePreview() {
    const el = document.getElementById('preview-text');
    if (el) el.textContent = trainingData.slice(0, 300) + (trainingData.length > 300 ? '...' : '');
  }

  // Load default
  loadPreset('shakespeare');

  // Temperature slider
  const tempSlider = document.getElementById('temperature');
  const tempVal = document.getElementById('temp-val');
  if (tempSlider) {
    tempSlider.addEventListener('input', () => {
      if (tempVal) tempVal.textContent = parseFloat(tempSlider.value).toFixed(1);
    });
  }

  // ===== N-gram Model =====
  function buildNgramModel(text, n) {
    const model = new Map();
    let uniqueChars = new Set();
    for (let i = 0; i <= text.length - n - 1; i++) {
      const context = text.substring(i, i + n);
      const next = text[i + n];
      uniqueChars.add(next);
      if (!model.has(context)) model.set(context, {});
      const dist = model.get(context);
      dist[next] = (dist[next] || 0) + 1;
    }
    return { model, vocabSize: uniqueChars.size, totalNgrams: text.length - n };
  }

  function sampleFromDist(dist, temperature) {
    const entries = Object.entries(dist);
    if (entries.length === 0) return null;
    if (entries.length === 1) return entries[0][0];

    // Apply temperature
    const total = entries.reduce((s, [, c]) => s + c, 0);
    const scaled = entries.map(([char, count]) => {
      const prob = count / total;
      return [char, Math.pow(prob, 1 / temperature)];
    });
    const scaledTotal = scaled.reduce((s, [, w]) => s + w, 0);

    let r = Math.random() * scaledTotal;
    for (const [char, weight] of scaled) {
      r -= weight;
      if (r <= 0) return char;
    }
    return scaled[scaled.length - 1][0];
  }

  function generate(model, seed, length, n, temperature) {
    let current = seed.slice(-n);

    // If seed is shorter than n, pad or find a matching key
    if (current.length < n) {
      // Try to find a key ending with the seed
      for (const key of model.keys()) {
        if (key.endsWith(current)) { current = key; break; }
      }
      if (current.length < n) {
        // Pick a random starting point
        const keys = Array.from(model.keys());
        current = keys[Math.floor(Math.random() * keys.length)];
      }
    }

    const chars = [];
    for (let i = 0; i < length; i++) {
      const dist = model.get(current);
      if (!dist) {
        // Backoff: try shorter context
        let found = false;
        for (let backoff = n - 1; backoff >= 1; backoff--) {
          const shorter = current.slice(-backoff);
          for (const [key, d] of model) {
            if (key.endsWith(shorter)) {
              const ch = sampleFromDist(d, temperature);
              if (ch) { chars.push(ch); current = current.slice(1) + ch; found = true; break; }
            }
          }
          if (found) break;
        }
        if (!found) break; // Dead end
      } else {
        const ch = sampleFromDist(dist, temperature);
        if (!ch) break;
        chars.push(ch);
        current = current.slice(1) + ch;
      }
    }
    return chars.join('');
  }

  // ===== UI =====
  function resetModel() {
    ngramModel = null;
    document.getElementById('train-step-4').style.display = 'none';
    document.getElementById('train-stats').style.display = 'none';
    const btn = document.getElementById('train-btn');
    btn.textContent = '⚡ Train Model';
    btn.disabled = false;
  }

  // Train button
  document.getElementById('train-btn').addEventListener('click', () => {
    if (trainingData.length < 200) {
      alert('Need at least 200 characters of training data.');
      return;
    }

    const orderEl = document.getElementById('ngram-order');
    order = parseInt(orderEl.value);

    const t0 = performance.now();
    const result = buildNgramModel(trainingData, order);
    const elapsed = (performance.now() - t0).toFixed(0);
    ngramModel = result.model;

    const btn = document.getElementById('train-btn');
    btn.textContent = '✅ Model Ready!';
    btn.disabled = true;

    const stats = document.getElementById('train-stats');
    stats.style.display = 'block';
    stats.innerHTML = `Built in <strong>${elapsed}ms</strong> · <strong>${result.model.size.toLocaleString()}</strong> unique ${order}-grams · <strong>${result.vocabSize}</strong> unique characters · <strong>${trainingData.length.toLocaleString()}</strong> chars of training data`;

    document.getElementById('train-step-4').style.display = 'block';
    document.getElementById('gen-prompt').focus();
  });

  // Generate on Enter
  const genPrompt = document.getElementById('gen-prompt');
  const genOutput = document.getElementById('gen-output');
  const genAgainBtn = document.getElementById('gen-again-btn');

  function doGenerate() {
    if (!ngramModel || isGenerating) return;
    const prompt = genPrompt.value;
    if (!prompt.trim()) return;

    const temp = tempSlider ? parseFloat(tempSlider.value) : 0.8;
    const output = generate(ngramModel, prompt, 300, order, temp);

    // Typewriter effect
    isGenerating = true;
    genOutput.innerHTML = `<span class="gen-prompt-echo">${escapeHtml(prompt)}</span><span class="gen-continuation"></span>`;
    const cont = genOutput.querySelector('.gen-continuation');
    let i = 0;
    function typeNext() {
      if (i < output.length) {
        cont.textContent += output[i];
        i++;
        setTimeout(typeNext, 18);
      } else {
        isGenerating = false;
        genAgainBtn.style.display = 'inline-block';
      }
    }
    genAgainBtn.style.display = 'none';
    typeNext();
  }

  genPrompt.addEventListener('keydown', e => {
    if (e.key === 'Enter') doGenerate();
  });
  genAgainBtn.addEventListener('click', doGenerate);

  // Share
  document.getElementById('share-btn').addEventListener('click', async () => {
    if (!ngramModel) return;
    try {
      // Serialize model as object
      const serialized = {};
      for (const [key, dist] of ngramModel) {
        serialized[key] = dist;
      }
      const payload = {
        type: 'ngram',
        order,
        model: serialized,
        preset: activePreset,
        dataLength: trainingData.length
      };

      const resp = await fetch('/api/models/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error('Save failed');
      const result = await resp.json();
      const shareLink = document.getElementById('share-link');
      const url = `${window.location.origin}/model.html?id=${result.id}`;
      shareLink.innerHTML = `Share link: <a href="${url}" target="_blank">${url}</a>`;
      shareLink.style.display = 'block';
    } catch (e) {
      console.error('Share failed:', e);
      alert('Failed to share. Try again.');
    }
  });

  function escapeHtml(t) {
    return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ===== Markov backend completion =====
  let markovDebounce = null;
  const markovInput = document.getElementById('markov-input');
  if (markovInput) {
    markovInput.addEventListener('input', () => {
      clearTimeout(markovDebounce);
      markovDebounce = setTimeout(fetchCompletion, 500);
    });
  }

  async function fetchCompletion() {
    const input = document.getElementById('markov-input');
    const output = document.getElementById('markov-output');
    const preset = document.getElementById('markov-preset').value;
    const text = input.value;
    if (!text.trim()) { output.innerHTML = ''; return; }
    try {
      const resp = await fetch(`/api/complete?text=${encodeURIComponent(text)}&preset=${preset}&length=100`);
      if (!resp.ok) { output.innerHTML = '<span style="color:#ef4444;">Rate limited. Wait a moment.</span>'; return; }
      const data = await resp.json();
      output.innerHTML = `<span>${escapeHtml(text)}</span><span style="color:var(--accent);font-weight:500;">${escapeHtml(data.text)}</span>`;
    } catch (e) {
      output.innerHTML = '';
    }
  }

})();
