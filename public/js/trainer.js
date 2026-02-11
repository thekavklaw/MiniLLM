// MiniLLM — Interactive Text Classifier + Markov Completion
(function() {
  'use strict';

  // ===== PRESETS =====
  const PRESETS = {
    sentiment: {
      categories: ['Happy 😊', 'Sad 😢'],
      examples: {
        'Happy 😊': [
          "I'm having the best day ever!", "This makes me so happy", "I love spending time with friends",
          "What a wonderful surprise", "I feel so grateful today", "Everything is going great",
          "This is absolutely amazing", "I can't stop smiling", "Today was perfect in every way",
          "I'm so excited about this", "Life is beautiful", "I feel fantastic right now",
          "This party is so much fun", "I got the promotion I wanted", "My heart is full of joy",
          "What a gorgeous day outside", "I'm proud of what I accomplished", "This food is delicious",
          "Spending time with family makes me happy", "I just love this song so much"
        ],
        'Sad 😢': [
          "I feel so lonely today", "Nothing seems to go right", "I miss my old friends",
          "This makes me want to cry", "I'm feeling really down", "Everything feels hopeless",
          "I can't stop feeling sad", "Today was the worst day", "I feel empty inside",
          "Nobody understands how I feel", "I'm so disappointed", "Life feels meaningless right now",
          "I lost something very important to me", "I feel like giving up", "The world seems so dark",
          "I can't find any motivation", "I'm heartbroken", "Everything reminds me of what I lost",
          "I just want to be alone and cry", "Nothing makes me happy anymore"
        ]
      }
    },
    spam: {
      categories: ['Spam 🚫', 'Not Spam ✅'],
      examples: {
        'Spam 🚫': [
          "CONGRATULATIONS! You won a free iPhone!", "Click here to claim your prize money",
          "Make $5000 per day working from home", "Buy cheap pills online now",
          "You have been selected as a winner", "Free gift card waiting for you",
          "Act now! Limited time offer expires today", "Enlarge your bank account instantly",
          "Nigerian prince needs your help transferring money", "You won the lottery! Send your details",
          "Hot singles in your area want to meet", "Discount medications shipped overnight",
          "Your account has been compromised click here", "Make money fast with this one weird trick",
          "FREE FREE FREE no credit card needed", "Congratulations you are our 1 millionth visitor",
          "Double your investment guaranteed no risk", "Urgent: verify your bank details now",
          "Amazing weight loss secret doctors hate", "Get rich quick with crypto secrets"
        ],
        'Not Spam ✅': [
          "Hey, are we still meeting for lunch?", "The project deadline is next Friday",
          "Can you review my pull request?", "Meeting moved to 3pm tomorrow",
          "Here are the notes from today's standup", "Happy birthday! Hope you have a great day",
          "The weather looks nice this weekend", "Did you see the game last night?",
          "I attached the quarterly report", "Let me know when you're free to chat",
          "Thanks for sending that over", "The new feature is ready for testing",
          "Can you pick up some groceries on the way home?", "I'll be working from home tomorrow",
          "Great job on the presentation today", "The kids have soccer practice at 4",
          "Here's the recipe you asked about", "Flight confirmation for next Tuesday",
          "Reminder: dentist appointment Thursday 2pm", "The package was delivered this morning"
        ]
      }
    },
    catdog: {
      categories: ['Cat 🐱', 'Dog 🐶'],
      examples: {
        'Cat 🐱': [
          "It purrs when you pet it softly", "Sleeps in a sunny spot by the window all day",
          "Knocks things off tables for fun", "Very independent and does its own thing",
          "Uses a litter box and grooms itself", "Loves to chase laser pointers around the room",
          "Climbs up the curtains and gets stuck", "Hisses when strangers come near",
          "Curls up in a tiny ball on the couch", "Brings you dead mice as presents",
          "Has retractable claws and whiskers", "Meows loudly at 3am for no reason",
          "Fits perfectly in a cardboard box", "Ignores you completely then wants attention",
          "Stares at birds through the window for hours"
        ],
        'Dog 🐶': [
          "Wags its tail excitedly when you come home", "Loves to play fetch in the park",
          "Barks at the mailman every single day", "Goes on walks and sniffs everything",
          "Loyal companion that follows you everywhere", "Rolls over for belly rubs",
          "Chews on shoes and furniture when bored", "Jumps up to greet visitors at the door",
          "Pants with tongue out after running around", "Digs holes in the backyard constantly",
          "Loves swimming and getting muddy", "Howls when it hears sirens outside",
          "Trained to sit stay and shake hands", "Drools when it sees food on the table",
          "Protects the house and barks at strangers"
        ]
      }
    }
  };

  let currentPreset = 'sentiment';
  let categories = [];
  let examples = {};  // { category: [text, ...] }
  let model = null;
  let vocab = {};     // word -> index
  let vocabSize = 0;

  // ===== INIT =====
  function init() {
    document.querySelectorAll('#train-step-1 .preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#train-step-1 .preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentPreset = btn.dataset.preset;
        loadPreset(currentPreset);
      });
    });

    document.getElementById('add-example-btn').addEventListener('click', addExample);
    document.getElementById('add-example-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') addExample();
    });
    document.getElementById('train-btn').addEventListener('click', trainModel);
    document.getElementById('test-input')?.addEventListener('input', classify);
    document.getElementById('share-btn')?.addEventListener('click', shareModel);

    // Custom category setup
    document.getElementById('add-cat-btn')?.addEventListener('click', addCustomCategory);
    document.getElementById('custom-cat-input')?.addEventListener('keydown', e => {
      if (e.key === 'Enter') addCustomCategory();
    });

    loadPreset('sentiment');
    initMarkov();
  }

  function loadPreset(name) {
    const customSetup = document.getElementById('custom-categories-setup');
    if (name === 'custom') {
      categories = [];
      examples = {};
      customSetup.style.display = 'block';
      renderExamples();
      updateCategorySelect();
      resetTrainState();
      return;
    }
    customSetup.style.display = 'none';
    const preset = PRESETS[name];
    if (!preset) return;
    categories = [...preset.categories];
    examples = {};
    categories.forEach(cat => {
      examples[cat] = [...preset.examples[cat]];
    });
    renderExamples();
    updateCategorySelect();
    resetTrainState();
  }

  function addCustomCategory() {
    const input = document.getElementById('custom-cat-input');
    const name = input.value.trim();
    if (!name || categories.includes(name)) return;
    categories.push(name);
    examples[name] = [];
    input.value = '';
    renderExamples();
    updateCategorySelect();
  }

  function updateCategorySelect() {
    const sel = document.getElementById('add-example-cat');
    sel.innerHTML = categories.map(c => `<option value="${c}">${c}</option>`).join('');
  }

  function addExample() {
    const input = document.getElementById('add-example-input');
    const cat = document.getElementById('add-example-cat').value;
    const text = input.value.trim();
    if (!text || !cat) return;
    if (!examples[cat]) examples[cat] = [];
    examples[cat].push(text);
    input.value = '';
    renderExamples();
    resetTrainState();
  }

  function removeExample(cat, idx) {
    examples[cat].splice(idx, 1);
    renderExamples();
    resetTrainState();
  }

  function renderExamples() {
    const container = document.getElementById('category-columns');
    container.innerHTML = categories.map(cat => `
      <div class="category-column">
        <div class="category-header">${cat} <span class="example-count">(${(examples[cat] || []).length})</span></div>
        <div class="example-list">
          ${(examples[cat] || []).map((ex, i) => `
            <div class="example-item">
              <span class="example-text">${escapeHtml(ex)}</span>
              <button class="example-remove" data-cat="${cat}" data-idx="${i}">×</button>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.example-remove').forEach(btn => {
      btn.addEventListener('click', () => removeExample(btn.dataset.cat, parseInt(btn.dataset.idx)));
    });
  }

  function escapeHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function resetTrainState() {
    model = null;
    vocab = {};
    vocabSize = 0;
    document.getElementById('train-step-4').style.display = 'none';
    document.getElementById('train-stats').style.display = 'none';
    const btn = document.getElementById('train-btn');
    btn.textContent = '⚡ Train (instant)';
    btn.disabled = false;
  }

  // ===== BAG OF WORDS =====
  function tokenize(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(w => w.length > 0);
  }

  function buildVocab() {
    vocab = {};
    let idx = 0;
    categories.forEach(cat => {
      (examples[cat] || []).forEach(text => {
        tokenize(text).forEach(word => {
          if (!(word in vocab)) vocab[word] = idx++;
        });
      });
    });
    vocabSize = idx;
  }

  function textToVector(text) {
    const vec = new Float32Array(vocabSize);
    tokenize(text).forEach(word => {
      if (word in vocab) vec[vocab[word]] = 1;
    });
    return vec;
  }

  // ===== TRAIN =====
  async function trainModel() {
    if (categories.length < 2) { alert('Need at least 2 categories.'); return; }
    const totalExamples = categories.reduce((s, c) => s + (examples[c] || []).length, 0);
    if (totalExamples < 4) { alert('Need at least 4 examples total.'); return; }

    const btn = document.getElementById('train-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Training...';

    buildVocab();

    // Prepare data
    const xs = [];
    const ys = [];
    categories.forEach((cat, catIdx) => {
      (examples[cat] || []).forEach(text => {
        xs.push(Array.from(textToVector(text)));
        const label = new Array(categories.length).fill(0);
        label[catIdx] = 1;
        ys.push(label);
      });
    });

    const xTensor = tf.tensor2d(xs);
    const yTensor = tf.tensor2d(ys);

    // Build model
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [vocabSize], units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: categories.length, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const result = await model.fit(xTensor, yTensor, {
      epochs: 50,
      batchSize: xs.length,
      shuffle: true,
      verbose: 0
    });

    xTensor.dispose();
    yTensor.dispose();

    const finalLoss = result.history.loss[result.history.loss.length - 1];
    const finalAcc = result.history.acc[result.history.acc.length - 1];

    btn.textContent = '✅ Trained!';
    const stats = document.getElementById('train-stats');
    stats.style.display = 'block';
    stats.innerHTML = `Trained on <strong>${totalExamples}</strong> examples · <strong>${vocabSize}</strong> word vocabulary · Loss: ${finalLoss.toFixed(4)} · Acc: ${(finalAcc * 100).toFixed(0)}%`;

    document.getElementById('train-step-4').style.display = 'block';
    renderConfidenceBars([]);

    // Focus test input
    document.getElementById('test-input').focus();
  }

  // ===== CLASSIFY =====
  function classify() {
    if (!model) return;
    const text = document.getElementById('test-input').value;
    if (!text.trim()) { renderConfidenceBars([]); return; }

    const vec = textToVector(text);
    const pred = model.predict(tf.tensor2d([Array.from(vec)]));
    const probs = pred.dataSync();
    pred.dispose();

    const results = categories.map((cat, i) => ({ category: cat, confidence: probs[i] }));
    results.sort((a, b) => b.confidence - a.confidence);
    renderConfidenceBars(results);
  }

  function renderConfidenceBars(results) {
    const container = document.getElementById('confidence-bars');
    if (!results.length) {
      container.innerHTML = '<div style="color:var(--text-dim);font-size:0.9rem;padding:1rem 0;">Type something above to see classification results...</div>';
      return;
    }
    container.innerHTML = results.map(r => `
      <div class="confidence-row">
        <div class="confidence-label">${r.category}</div>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" style="width:${(r.confidence * 100).toFixed(1)}%"></div>
        </div>
        <div class="confidence-pct">${(r.confidence * 100).toFixed(1)}%</div>
      </div>
    `).join('');
  }

  // ===== SHARE =====
  async function shareModel() {
    if (!model) return;
    try {
      const saveData = await new Promise((resolve, reject) => {
        model.save(tf.io.withSaveHandler(async (artifacts) => {
          resolve({
            topology: artifacts.modelTopology,
            weightSpecs: artifacts.weightSpecs,
            weightData: Array.from(new Uint8Array(artifacts.weightData)),
            vocab,
            categories,
            examples,
            preset: currentPreset
          });
          return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
        }));
      });

      const resp = await fetch('/api/models/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(saveData)
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
  }

  // ===== MARKOV COMPLETION =====
  let markovDebounce = null;

  function initMarkov() {
    const input = document.getElementById('markov-input');
    if (!input) return;
    input.addEventListener('input', () => {
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

  // ===== BOOT =====
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
