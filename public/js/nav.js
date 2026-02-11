/**
 * Navigation, progress tracking, chapter transitions
 * @module Nav
 */
'use strict';

const Nav = {
  chapters: [
    { id: 1, title: 'What IS a Neural Network?', path: '/chapters/1-neuron.html', icon: '🧠' },
    { id: 2, title: 'Neurons That Learn', path: '/chapters/2-learning.html', icon: '📈' },
    { id: 3, title: 'Layers of Thinking', path: '/chapters/3-layers.html', icon: '🔗' },
    { id: 4, title: 'The Playground', path: '/chapters/4-playground.html', icon: '🎮' },
    { id: 5, title: 'How Computers Read Words', path: '/chapters/5-words.html', icon: '📝' },
    { id: 6, title: 'Paying Attention', path: '/chapters/6-attention.html', icon: '👁️' },
    { id: 7, title: 'The Transformer', path: '/chapters/7-transformer.html', icon: '⚡' },
    { id: 8, title: 'From Tiny to GPT', path: '/chapters/8-scale.html', icon: '🚀' }
  ],

  getProgress() {
    try {
      return JSON.parse(localStorage.getItem('minillm-progress') || '{}');
    } catch { return {}; }
  },

  setCompleted(chapterId) {
    const p = this.getProgress();
    p[chapterId] = { completed: true, timestamp: Date.now() };
    localStorage.setItem('minillm-progress', JSON.stringify(p));
  },

  isCompleted(chapterId) {
    return !!this.getProgress()[chapterId]?.completed;
  },

  getCompletedCount() {
    return Object.values(this.getProgress()).filter(v => v.completed).length;
  },

  /** Inject navigation bar into current page */
  injectNav(currentChapter = null) {
    const nav = document.createElement('nav');
    nav.className = 'mini-nav';
    nav.innerHTML = `
      <a href="/" class="nav-logo">
        <span class="nav-logo-icon">🧠</span>
        <span class="nav-logo-text">MiniLLM</span>
      </a>
      <div class="nav-chapters">
        ${this.chapters.map(ch => `
          <a href="${ch.path}" class="nav-chapter ${ch.id === currentChapter ? 'active' : ''} ${this.isCompleted(ch.id) ? 'completed' : ''}" title="${ch.title}">
            <span class="nav-ch-num">${ch.id}</span>
          </a>
        `).join('')}
      </div>
      <div class="nav-progress">
        <span>${this.getCompletedCount()}/8</span>
      </div>
    `;
    document.body.prepend(nav);

    // Add bottom navigation for chapters
    if (currentChapter) {
      const prev = currentChapter > 1 ? this.chapters[currentChapter - 2] : null;
      const next = currentChapter < 8 ? this.chapters[currentChapter] : null;
      const bottomNav = document.createElement('div');
      bottomNav.className = 'chapter-bottom-nav';
      bottomNav.innerHTML = `
        ${prev ? `<a href="${prev.path}" class="btn-chapter-nav">← ${prev.title}</a>` : '<span></span>'}
        ${next ? `<a href="${next.path}" class="btn-chapter-nav">${next.title} →</a>` : '<span></span>'}
      `;
      document.querySelector('.chapter-content')?.appendChild(bottomNav);
    }
  }
};

if (typeof window !== 'undefined') {
  window.Nav = Nav;
  document.addEventListener('DOMContentLoaded', () => {
    // Auto-inject nav if data-chapter attribute exists
    const chAttr = document.body.dataset.chapter;
    if (chAttr) Nav.injectNav(parseInt(chAttr));
    else if (document.body.dataset.page === 'home') Nav.injectNav();
  });
}
