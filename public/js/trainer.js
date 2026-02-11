// MiniLLM — Neural Network Language Model (backend LSTM inference)
(function() {
  'use strict';

  // Turnstile helper
  const TURNSTILE_SITEKEY = '0x4AAAAAACZGxpcf0vhl9Oes';
  let turnstileToken = null;
  let turnstileWidgetId = null;

  function getTurnstileToken() {
    return new Promise((resolve) => {
      if (typeof turnstile === 'undefined') { resolve(null); return; }
      const container = document.getElementById('turnstile-container');
      if (!container) { resolve(null); return; }
      container.style.opacity = '1';
      container.style.pointerEvents = 'auto';
      
      if (turnstileWidgetId !== null) {
        try { turnstile.reset(turnstileWidgetId); } catch(e) {}
      }
      
      turnstileWidgetId = turnstile.render(container, {
        sitekey: TURNSTILE_SITEKEY,
        callback: function(token) {
          turnstileToken = token;
          container.style.opacity = '0';
          container.style.pointerEvents = 'none';
          resolve(token);
        },
        'error-callback': function() {
          container.style.opacity = '0';
          container.style.pointerEvents = 'none';
          resolve(null);
        }
      });
    });
  }

  // Collapsible sections (landing page doesn't load nav.js)
  document.querySelectorAll('.collapsible-header').forEach(header => {
    if (header._collapsibleBound) return;
    header._collapsibleBound = true;
    header.addEventListener('click', () => {
      header.classList.toggle('open');
      let body = header.nextElementSibling;
      if (!body || !body.classList.contains('collapsible-body')) {
        body = header.parentElement.querySelector('.collapsible-body');
      }
      if (body) body.classList.toggle('open');
    });
  });

  let activePreset = 'shakespeare';
  let isGenerating = false;
  let modelReady = false;
  let isCustomTrained = false;
  let customModelToken = null;

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

      // Reset model state when switching presets
      resetModel();
    });
  });

  function resetModel() {
    modelReady = false;
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
      trainBtn.disabled = false;
      trainBtn.textContent = '⚡ Load Neural Network';
    }
    const dashboard = document.getElementById('training-dashboard');
    if (dashboard) { dashboard.style.display = 'none'; dashboard.innerHTML = ''; }
    const sampleEl = document.getElementById('sample-text');
    if (sampleEl) { sampleEl.style.display = 'none'; sampleEl.innerHTML = ''; }
    const commentary = document.getElementById('output-commentary');
    if (commentary) commentary.style.display = 'none';
    const step4 = document.getElementById('train-step-4');
    if (step4) step4.style.display = 'none';
    const genOutput = document.getElementById('gen-output');
    if (genOutput) genOutput.innerHTML = '';
    const shareLink = document.getElementById('share-link');
    if (shareLink) shareLink.style.display = 'none';
    const shareBtn = document.getElementById('share-btn');
    if (shareBtn) { shareBtn.disabled = false; shareBtn.textContent = '📤 Share'; }
    const again = document.getElementById('gen-again-btn');
    if (again) again.style.display = 'none';
  }

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
        const customArea = document.getElementById('custom-text');
        const customText = customArea ? customArea.value.trim() : '';
        if (customText.length < 100) {
          alert('Paste at least 100 characters of text to train on.');
          return;
        }
        trainBtn.disabled = true;
        trainBtn.textContent = '🔒 Verifying...';
        try {
          const tsToken = await getTurnstileToken();
          if (!tsToken) { alert('Verification failed. Please try again.'); resetModel(); return; }
          trainBtn.textContent = '🧠 Training your model (~15-30s)...';
          const resp = await fetch('/api/train-custom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: customText, turnstileToken: tsToken })
          });
          const data = await resp.json();
          if (!resp.ok) { alert(data.error || 'Training failed'); resetModel(); return; }

          // Show dashboard
          const dashboard = document.getElementById('training-dashboard');
          if (dashboard) {
            dashboard.style.display = 'block';
            dashboard.innerHTML = `
              <div class="train-stats">
                <div class="stat-item"><div class="stat-value">LSTM</div><div class="stat-label">Architecture</div></div>
                <div class="stat-item"><div class="stat-value">${data.totalParams.toLocaleString()}</div><div class="stat-label">Parameters</div></div>
                <div class="stat-item"><div class="stat-value">${data.epochs}</div><div class="stat-label">Epochs</div></div>
                <div class="stat-item"><div class="stat-value">${data.finalLoss.toFixed(3)}</div><div class="stat-label">Final Loss</div></div>
                <div class="stat-item"><div class="stat-value">${(data.trainTimeMs/1000).toFixed(1)}s</div><div class="stat-label">Train Time</div></div>
                <div class="stat-item"><div class="stat-value">${data.vocabSize}</div><div class="stat-label">Vocab Size</div></div>
              </div>`;
          }

          // Generate sample
          const genResp = await fetch('/api/generate-custom', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: customText.slice(0, 20), temperature: 0.7, length: 150, modelToken: data.modelToken })
          });
          const genData = await genResp.json();
          const sample = document.getElementById('sample-text');
          if (sample && genData.text) {
            sample.style.display = 'block';
            sample.innerHTML = `<div class="gen-output"><span class="seed-text">${escapeHtml(customText.slice(0,20))}</span><span class="gen-text">${escapeHtml(genData.text)}</span></div>`;
          }

          // Show commentary
          const commentary = document.getElementById('output-commentary');
          if (commentary) {
            commentary.style.display = 'block';
            commentary.innerHTML = `<strong>What happened:</strong> We just trained a real LSTM neural network with ${data.totalParams.toLocaleString()} parameters on your text. Loss dropped to ${data.finalLoss.toFixed(3)} after ${data.epochs} epochs. The output is rough — with only ${data.samples} training samples and ${data.totalParams.toLocaleString()} parameters, this model is trying to learn language from almost nothing. GPT-4 has 36 <em>billion</em> times more parameters and trained on trillions of words.`;
          }

          // Show generate step
          isCustomTrained = true;
          modelReady = true;
          customModelToken = data.modelToken;
          const step4 = document.getElementById('train-step-4');
          if (step4) step4.style.display = 'block';
          trainBtn.textContent = '✅ Trained! Try generating below';
          trainBtn.disabled = true;
        } catch(e) {
          alert('Training failed: ' + e.message);
          resetModel();
        }
        return;
      }

      trainBtn.disabled = true;
      trainBtn.textContent = '🔒 Verifying...';

      try {
        const tsToken = await getTurnstileToken();
        if (!tsToken) { alert('Verification failed. Please try again.'); resetModel(); return; }
        trainBtn.textContent = '🧠 Loading neural network...';

        const infoResp = await fetch('/api/model-info');
        const info = await infoResp.json();

        if (!info.models.includes(activePreset)) {
          trainBtn.textContent = '❌ Model not available';
          setTimeout(resetModel, 2000);
          return;
        }

        // Generate sample
        const t0 = performance.now();
        const testResp = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: activePreset === 'python' ? 'def ' : 'The ', preset: activePreset, temperature: 0.7, length: 150, turnstileToken: tsToken })
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
                <div class="stat-label">Inference time</div>
              </div>
            </div>
            <div class="train-explanation">
              <p>This is a <strong>real neural network</strong> — an LSTM (Long Short-Term Memory) with 35,900 trained parameters. 
              It learned to predict the next character by reading through training text thousands of times, adjusting its weights each pass. 
              ChatGPT works on the exact same principle: <em>given some text, predict what comes next.</em> 
              The difference is scale — GPT-4 has <strong>1.8 trillion</strong> parameters (50 million times more than this model).</p>
              <p style="margin-top:0.7rem; font-size:0.85rem;">
              <strong>Compute comparison:</strong> This tiny LSTM trained in about 2 minutes on a single CPU core and uses ~140KB of memory. 
              GPT-4 reportedly took <strong>$100+ million</strong> of compute to train across thousands of GPUs running for months.
              The same fundamental math — just at a completely different scale.</p>
            </div>
          `;
        }

        // Show sample with commentary
        const sampleEl = document.getElementById('sample-text');
        if (sampleEl && testData.text) {
          const seedText = activePreset === 'python' ? 'def ' : 'The ';
          sampleEl.style.display = 'block';
          sampleEl.innerHTML = `<strong>Sample output:</strong><br><span class="seed-text">${escapeHtml(seedText)}</span><span class="gen-text">${escapeHtml(testData.text)}</span>`;
        }

        // Show commentary
        const commentary = document.getElementById('output-commentary');
        if (commentary) {
          commentary.style.display = 'block';
          const presetCommentary = {
            shakespeare: `Notice how the output <em>almost</em> sounds like Shakespeare — it picked up patterns like rhyming couplets, 
              "thee" and "thy," and iambic-ish rhythm. But look closer: many "words" are gibberish ("thith," "fant," "shoml"). 
              The network learned that certain letter combinations are common in Shakespeare, but with only 35,900 parameters and 15KB of training text, 
              it can't learn actual English vocabulary or grammar. <strong>GPT-4 doesn't have this problem</strong> — trained on 13 trillion tokens 
              (roughly 50 million books), it has enough capacity to learn every word, every grammatical rule, and even subtle things like sarcasm and humor.
              That's the magic of scale: the same "predict the next character" algorithm, but with 50 million times more parameters and billions of times more data.`,
            recipes: `The output looks recipe-ish — you might spot fragments like "Cook for," "minutes," "the tomato." 
              But the instructions don't make sense: quantities are wrong, steps are garbled, ingredients appear randomly. 
              Our network has just enough capacity to learn that recipe text contains words like "cup," "heat," and "minutes" — 
              but not enough to understand what a recipe actually <em>is</em>. <strong>GPT-4, with 1.8 trillion parameters,</strong> 
              can write a complete, coherent recipe from scratch because it's seen millions of real recipes and understands 
              the structure: ingredients list → prep steps → cooking steps → serving suggestion. Same algorithm, vastly different capability.`,
            python: `You might see fragments that look code-like: "def," parentheses, indentation patterns. 
              But the "code" is nonsensical — variable names are random characters, function bodies don't compute anything meaningful. 
              With only 2,600 characters of training data and 35,900 parameters, the model learned Python's <em>syntax patterns</em> 
              (indentation, colons, parentheses) but has zero understanding of what code <em>does</em>. 
              <strong>GPT-4 can write working programs</strong> because it trained on billions of lines of real code and learned 
              not just syntax, but logic, algorithms, and debugging strategies. The gap between our toy model and GPT is like 
              the gap between a toddler scribbling and a novelist writing a book — same tool (language), incomparable skill.`
          };
          commentary.innerHTML = presetCommentary[activePreset] || '';
        }

        trainBtn.textContent = '✅ Neural network loaded!';
        modelReady = true;

        // Show generate section
        const step4 = document.getElementById('train-step-4');
        if (step4) step4.style.display = 'block';

      } catch (e) {
        console.error(e);
        trainBtn.textContent = '❌ Failed to connect';
        setTimeout(resetModel, 2000);
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
        const endpoint = (activePreset === 'custom' && isCustomTrained) ? '/api/generate-custom' : '/api/generate';
        const body = (activePreset === 'custom' && isCustomTrained) 
          ? { prompt, temperature: temp, length: 200, modelToken: customModelToken }
          : { prompt, preset: activePreset, temperature: temp, length: 200 };
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
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
      shareBtn.textContent = '⏳ Saving...';

      try {
        const temp = tempSlider ? parseFloat(tempSlider.value) : 0.7;
        // Generate fresh output for sharing
        const resp = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: genInput ? genInput.value || 'The ' : 'The ', preset: activePreset, temperature: temp, length: 300 })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        const saveResp = await fetch('/api/models/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            type: 'lstm-output',
            preset: activePreset,
            prompt: genInput ? genInput.value || 'The ' : 'The ',
            output: data.text,
            temperature: temp,
            model: { architecture: 'LSTM', params: 35900, units: 64 }
          })
        });

        if (!saveResp.ok) {
          const err = await saveResp.json().catch(() => ({}));
          throw new Error(err.error || 'Save failed');
        }
        const saveData = await saveResp.json();

        const shareLink = document.getElementById('share-link');
        if (shareLink && saveData.id) {
          const url = `${window.location.origin}/shared.html?id=${saveData.id}`;
          shareLink.innerHTML = `<strong>Share link:</strong> <a href="${url}" target="_blank">${url}</a>`;
          shareLink.style.display = 'block';
        }
        shareBtn.textContent = '✅ Shared!';
        setTimeout(() => { shareBtn.textContent = '📤 Share'; shareBtn.disabled = false; }, 3000);
      } catch (e) {
        console.error('Share error:', e);
        shareBtn.textContent = '❌ ' + (e.message || 'Failed');
        setTimeout(() => { shareBtn.textContent = '📤 Share'; shareBtn.disabled = false; }, 2000);
      }
    });
  }

  // ===== Markov completion (backend bonus section) =====
  const markovInput = document.getElementById('markov-input');
  const markovOutput = document.getElementById('markov-output');
  const markovPreset = document.getElementById('markov-preset');
  const chainViz = document.getElementById('markov-chain-viz');
  let markovTimer = null;

  function renderChainViz(seed, generated) {
    if (!chainViz) return;
    chainViz.innerHTML = '';
    // Show the chain: each character as a small block, seed in dim, generated colored by confidence
    const allChars = seed + generated;
    const seedLen = seed.length;
    // Only show last 60 chars to keep it manageable
    const start = Math.max(0, allChars.length - 80);
    const visible = allChars.slice(start);
    const seedVisible = Math.max(0, seedLen - start);

    for (let i = 0; i < visible.length; i++) {
      const ch = visible[i];
      const isSeed = i < seedVisible;
      const span = document.createElement('span');
      span.textContent = ch === ' ' ? '·' : ch === '\n' ? '↵' : ch === '\r' ? '' : ch;
      if (!span.textContent) continue;
      span.style.cssText = `
        display:inline-block; padding:2px 3px; font-family:monospace; font-size:0.75rem; 
        border-radius:3px; line-height:1.4;
        ${isSeed 
          ? 'background:rgba(0,0,0,0.05); color:#94a3b8;' 
          : `background:rgba(139,92,246,${0.08 + Math.random()*0.15}); color:var(--accent); font-weight:500;`
        }
      `;
      chainViz.appendChild(span);

      // Add arrow every 5 generated chars
      if (!isSeed && (i - seedVisible) > 0 && (i - seedVisible) % 8 === 0) {
        const arrow = document.createElement('span');
        arrow.textContent = '→';
        arrow.style.cssText = 'color:#d4d4d8; font-size:0.6rem; margin:0 1px;';
        chainViz.appendChild(arrow);
      }
    }
  }

  if (markovInput) {
    markovInput.addEventListener('input', () => {
      clearTimeout(markovTimer);
      markovTimer = setTimeout(async () => {
        const text = markovInput.value;
        if (text.length < 3) { 
          if (markovOutput) markovOutput.textContent = ''; 
          if (chainViz) chainViz.innerHTML = '';
          return; 
        }
        const preset = markovPreset ? markovPreset.value : 'shakespeare';
        try {
          const resp = await fetch(`/api/complete?text=${encodeURIComponent(text)}&preset=${preset}&length=150`);
          if (!resp.ok) return;
          const data = await resp.json();
          if (markovOutput) markovOutput.innerHTML = `<span class="seed-text">${escapeHtml(text)}</span><span class="gen-text">${escapeHtml(data.text)}</span>`;
          renderChainViz(text, data.text);
        } catch (e) { /* ignore */ }
      }, 400);
    });
  }
})();
