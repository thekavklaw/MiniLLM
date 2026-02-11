// MiniLLM — Neural Network Language Model (backend LSTM inference)
(function() {
  'use strict';

  let activePreset = 'shakespeare';
  let isGenerating = false;
  let modelReady = false;

  // ===== Preset buttons =====
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activePreset = btn.dataset.preset;

      const customArea = document.getElementById('custom-text');
      if (activePreset === 'custom') {
        if (customArea) customArea.style.display = 'block';
      } else {
        if (customArea) customArea.style.display = 'none';
        loadPreview(activePreset);
      }
    });
  });

  async function loadPreview(name) {
    const presets = { shakespeare: '/data/shakespeare.txt', recipes: '/data/recipes.txt', python: '/data/python.txt' };
    if (!presets[name]) return;
    try {
      const resp = await fetch(presets[name]);
      const text = await resp.text();
      const el = document.getElementById('preview-text');
      if (el) el.textContent = text.slice(0, 300) + (text.length > 300 ? '...' : '');
    } catch (e) { console.error(e); }
  }

  loadPreview('shakespeare');

  // Temperature slider
  const tempSlider = document.getElementById('temperature');
  const tempVal = document.getElementById('temp-val');
  if (tempSlider) {
    tempSlider.addEventListener('input', () => {
      if (tempVal) tempVal.textContent = parseFloat(tempSlider.value).toFixed(1);
    });
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ===== Train button (loads pre-trained LSTM from server) =====
  const trainBtn = document.getElementById('train-btn');
  if (trainBtn) {
    trainBtn.addEventListener('click', async () => {
      if (activePreset === 'custom') {
        alert('Custom text training happens in your browser. For the neural network demo, pick a preset — these models were pre-trained on the server.');
        return;
      }

      trainBtn.disabled = true;
      trainBtn.textContent = '🧠 Loading neural network...';

      try {
        // Verify model is loaded on server
        const infoResp = await fetch('/api/model-info');
        const info = await infoResp.json();

        if (!info.models.includes(activePreset)) {
          trainBtn.textContent = '❌ Model not available';
          return;
        }

        // Test generation
        const t0 = performance.now();
        const testResp = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: activePreset === 'python' ? 'def ' : 'The ', preset: activePreset, temperature: 0.7, length: 100 })
        });
        const testData = await testResp.json();
        const elapsed = (performance.now() - t0).toFixed(0);

        // Show dashboard
        const dashboard = document.getElementById('training-dashboard');
        if (dashboard) {
          dashboard.style.display = 'block';
          dashboard.innerHTML = `
            <div class="train-stats">
              <div class="stat-item">
                <div class="stat-value">LSTM</div>
                <div class="stat-label">Architecture</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">35,900</div>
                <div class="stat-label">Parameters</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">64</div>
                <div class="stat-label">Hidden units</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">${elapsed}ms</div>
                <div class="stat-label">Response time</div>
              </div>
            </div>
            <div class="train-explanation">
              <p>This is a <strong>real neural network</strong> — an LSTM (Long Short-Term Memory) with 35,900 trained parameters. 
              It learned to predict the next character by reading thousands of examples. 
              ChatGPT works on the exact same principle: <em>given text, predict what comes next.</em> 
              GPT-4 just does it with 1.8 trillion parameters instead of 35,900.</p>
            </div>
          `;
        }

        // Show sample
        const sampleEl = document.getElementById('sample-text');
        if (sampleEl && testData.text) {
          const seedText = activePreset === 'python' ? 'def ' : 'The ';
          sampleEl.style.display = 'block';
          sampleEl.innerHTML = `<strong>Sample output:</strong><br><span class="seed-text">${escapeHtml(seedText)}</span><span class="gen-text">${escapeHtml(testData.text)}</span>`;
        }

        trainBtn.textContent = '✅ Neural network loaded!';
        modelReady = true;

        // Show generate section
        const step4 = document.getElementById('train-step-4');
        if (step4) step4.style.display = 'block';

      } catch (e) {
        console.error(e);
        trainBtn.textContent = '❌ Failed to connect';
        setTimeout(() => { trainBtn.textContent = '⚡ Load Neural Network'; trainBtn.disabled = false; }, 2000);
      }
    });
  }

  // ===== Generate text =====
  const genInput = document.getElementById('gen-prompt');
  const genOutput = document.getElementById('gen-output');

  if (genInput) {
    genInput.addEventListener('keydown', async (e) => {
      if (e.key !== 'Enter' || isGenerating || !modelReady) return;
      e.preventDefault();
      
      const prompt = genInput.value;
      if (!prompt) return;

      isGenerating = true;
      genOutput.innerHTML = `<span class="seed-text">${escapeHtml(prompt)}</span><span class="gen-text gen-loading">▊</span>`;

      try {
        const temp = tempSlider ? parseFloat(tempSlider.value) : 0.7;
        const resp = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, preset: activePreset, temperature: temp, length: 200 })
        });
        const data = await resp.json();

        if (data.error) {
          genOutput.innerHTML = `<span style="color:#e57373;">${escapeHtml(data.error)}</span>`;
          isGenerating = false;
          return;
        }

        // Typewriter effect
        genOutput.innerHTML = `<span class="seed-text">${escapeHtml(prompt)}</span>`;
        const genSpan = document.createElement('span');
        genSpan.className = 'gen-text';
        genOutput.appendChild(genSpan);

        let i = 0;
        function type() {
          if (i < data.text.length) {
            genSpan.textContent += data.text[i];
            i++;
            setTimeout(type, 25);
          } else {
            isGenerating = false;
            const again = document.getElementById('gen-again-btn');
            if (again) again.style.display = 'inline-block';
          }
        }
        type();
      } catch (e) {
        genOutput.innerHTML = '<span style="color:#e57373;">Connection error. Try again.</span>';
        isGenerating = false;
      }
    });
  }

  // Generate again button
  const againBtn = document.getElementById('gen-again-btn');
  if (againBtn) {
    againBtn.addEventListener('click', () => {
      if (genInput && genInput.value) {
        genInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
      }
    });
  }

  // ===== Share =====
  const shareBtn = document.getElementById('share-btn');
  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      if (!modelReady) return;
      shareBtn.disabled = true;
      shareBtn.textContent = '⏳ Generating sharable output...';

      try {
        const temp = tempSlider ? parseFloat(tempSlider.value) : 0.7;
        const resp = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: genInput ? genInput.value || 'The ' : 'The ', preset: activePreset, temperature: temp, length: 300 })
        });
        const data = await resp.json();

        const saveResp = await fetch('/api/models/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'lstm-output',
            preset: activePreset,
            prompt: genInput ? genInput.value : 'The ',
            output: data.text,
            temperature: temp
          })
        });
        const saveData = await saveResp.json();

        const shareLink = document.getElementById('share-link');
        if (shareLink) {
          const url = `${window.location.origin}/model.html?id=${saveData.id}`;
          shareLink.innerHTML = `<strong>Share link:</strong> <a href="${url}" target="_blank">${url}</a>`;
          shareLink.style.display = 'block';
        }
        shareBtn.textContent = '📤 Shared!';
      } catch (e) {
        shareBtn.textContent = '❌ Failed';
        setTimeout(() => { shareBtn.textContent = '📤 Share'; shareBtn.disabled = false; }, 2000);
      }
    });
  }

  // ===== Markov completion (backend bonus section) =====
  const markovInput = document.getElementById('markov-input');
  const markovOutput = document.getElementById('markov-output');
  const markovPreset = document.getElementById('markov-preset');
  let markovTimer = null;

  if (markovInput) {
    markovInput.addEventListener('input', () => {
      clearTimeout(markovTimer);
      markovTimer = setTimeout(async () => {
        const text = markovInput.value;
        if (text.length < 3) { if (markovOutput) markovOutput.textContent = ''; return; }
        const preset = markovPreset ? markovPreset.value : 'shakespeare';
        try {
          const resp = await fetch(`/api/complete?text=${encodeURIComponent(text)}&preset=${preset}&length=150`);
          if (!resp.ok) return;
          const data = await resp.json();
          if (markovOutput) markovOutput.innerHTML = `<span class="seed-text">${escapeHtml(text)}</span><span class="gen-text">${escapeHtml(data.text)}</span>`;
        } catch (e) { /* ignore */ }
      }, 400);
    });
  }
})();
