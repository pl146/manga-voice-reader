(() => {
  const c = document.getElementById('stars');
  c.width = 640;
  c.height = 600;
  const ctx = c.getContext('2d');
  let w = 640, h = 600, stars = [];
  function init() {
    stars = [];
    for (let i = 0; i < 80; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 2.5 + 0.8,
        dx: (Math.random() - 0.5) * 0.5,
        dy: Math.random() * 0.4 + 0.15,
        o: Math.random() * 0.7 + 0.3,
      });
    }
  }
  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (const s of stars) {
      s.x += s.dx;
      s.y += s.dy;
      s.o += (Math.random() - 0.5) * 0.02;
      s.o = Math.max(0.1, Math.min(0.8, s.o));
      if (s.y > h) { s.y = 0; s.x = Math.random() * w; }
      if (s.x < 0) s.x = w;
      if (s.x > w) s.x = 0;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${s.o})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  init();
  draw();
})();
