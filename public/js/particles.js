/**
 * Floating particle background animation
 * @module Particles
 */
'use strict';

class ParticleSystem {
  constructor(canvasId = 'particles-canvas') {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.id = canvasId;
      this.canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;';
      document.body.prepend(this.canvas);
    }
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.connections = [];
    this.mouse = { x: -1000, y: -1000 };
    this.running = false;
    this.resize();
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('mousemove', e => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.initParticles();
  }

  initParticles() {
    const count = Math.min(60, Math.floor(this.canvas.width * this.canvas.height / 20000));
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 2.5 + 1,
        opacity: Math.random() * 0.4 + 0.1
      });
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.animate();
  }

  stop() {
    this.running = false;
  }

  animate() {
    if (!this.running) return;
    const { ctx, canvas, particles, mouse } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update and draw particles
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(139, 92, 246, ${p.opacity})`;
      ctx.fill();
    }

    // Draw connections
    const maxDist = 150;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          const alpha = (1 - dist / maxDist) * 0.15;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      // Mouse connection
      const dx = particles[i].x - mouse.x;
      const dy = particles[i].y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 200) {
        const alpha = (1 - dist / 200) * 0.2;
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

if (typeof window !== 'undefined') {
  window.ParticleSystem = ParticleSystem;
}
