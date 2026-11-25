/* ==========================================================================
   LLM Pipeline Explorer — Interactive Logic & 3D Embedding Engine
   ========================================================================== */

(() => {
  "use strict";

  // --------------------------------------------------------------------------
  // 1. Curated Datasets & Sample Sentences
  // --------------------------------------------------------------------------
  const SAMPLE_SENTENCES = {
    ai: "Large language models predict the next token based on context.",
    space: "Astrophysicists study cosmic microwave background radiation from distant galaxies.",
    food: "Fresh Italian pasta simmers slowly in rich San Marzano tomato sauce.",
    marine: "Dolphins use biological echolocation to navigate deep ocean currents.",
  };

  const CLUSTERS = {
    ai: { label: "AI & Computation", tag: "AI & Tech", color: "#38bdf8" },
    space: { label: "Space & Astronomy", tag: "Space", color: "#a855f7" },
    food: { label: "Culinary & Food", tag: "Culinary", color: "#fb923c" },
    marine: { label: "Marine Biology", tag: "Marine", color: "#10b981" },
    custom: { label: "Custom Inputs", tag: "Custom", color: "#f43f5e" },
  };

  const DATASET_SENTENCES = [
    {
      id: "ai-1",
      cluster: "ai",
      text: "Large language models predict the next token based on context.",
      vec: [0.82, -0.45, 0.56],
    },
    {
      id: "ai-2",
      cluster: "ai",
      text: "Deep neural networks learn representations from massive datasets.",
      vec: [0.76, -0.52, 0.49],
    },
    {
      id: "ai-3",
      cluster: "ai",
      text: "Transformers use multi-head self-attention to process language tokens.",
      vec: [0.89, -0.38, 0.62],
    },
    {
      id: "space-1",
      cluster: "space",
      text: "Astrophysicists study cosmic microwave radiation across galaxies.",
      vec: [-0.72, 0.68, 0.44],
    },
    {
      id: "space-2",
      cluster: "space",
      text: "Telescopes observe distant exoplanets orbiting binary star systems.",
      vec: [-0.81, 0.59, 0.35],
    },
    {
      id: "space-3",
      cluster: "space",
      text: "Planets and stars form inside dense swirling nebulae.",
      vec: [-0.65, 0.74, 0.51],
    },
    {
      id: "food-1",
      cluster: "food",
      text: "Fresh Italian pasta tastes best with basil and olive oil.",
      vec: [0.22, 0.79, -0.58],
    },
    {
      id: "food-2",
      cluster: "food",
      text: "Baking artisan sourdough bread requires patience and wild yeast.",
      vec: [0.14, 0.71, -0.66],
    },
    {
      id: "food-3",
      cluster: "food",
      text: "Chefs prepare gourmet dishes using seasonal farm ingredients.",
      vec: [0.29, 0.85, -0.49],
    },
    {
      id: "marine-1",
      cluster: "marine",
      text: "Sea turtles glide gracefully through warm tropical coral reefs.",
      vec: [-0.32, -0.78, -0.54],
    },
    {
      id: "marine-2",
      cluster: "marine",
      text: "Dolphins use echolocation to communicate in deep ocean waters.",
      vec: [-0.41, -0.71, -0.47],
    },
    {
      id: "marine-3",
      cluster: "marine",
      text: "Whales migrate thousands of miles across polar ocean basins.",
      vec: [-0.25, -0.84, -0.61],
    },
  ];

  // Token chip warm-editorial color palettes
  const TOKEN_PALETTE = [
    { bg: "rgba(234, 88, 12, 0.12)", border: "rgba(234, 88, 12, 0.38)", text: "#9a3412" },
    { bg: "rgba(15, 118, 110, 0.12)", border: "rgba(15, 118, 110, 0.38)", text: "#115e59" },
    { bg: "rgba(126, 34, 206, 0.11)", border: "rgba(126, 34, 206, 0.35)", text: "#6b21a8" },
    { bg: "rgba(2, 132, 199, 0.12)", border: "rgba(2, 132, 199, 0.38)", text: "#075985" },
    { bg: "rgba(190, 24, 93, 0.11)", border: "rgba(190, 24, 93, 0.35)", text: "#9d174d" },
    { bg: "rgba(161, 98, 7, 0.13)", border: "rgba(161, 98, 7, 0.38)", text: "#854d0e" },
  ];

  // --------------------------------------------------------------------------
  // 2. Theme Switcher
  // --------------------------------------------------------------------------
  const themeBtns = document.querySelectorAll(".theme-btn");
  themeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.themeTarget;
      document.documentElement.setAttribute("data-theme", target);
      themeBtns.forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  // --------------------------------------------------------------------------
  // 3. Step 1: Subword-Aware Tokenizer & Vocabulary Builder
  // --------------------------------------------------------------------------
  const tokenInput = document.getElementById("token-input");
  const tokensGrid = document.getElementById("tokens-grid");
  const tokenStatsBadge = document.getElementById("token-stats-badge");
  const tokenTooltip = document.getElementById("token-tooltip");
  const vocabDrawer = document.getElementById("vocab-drawer");
  const vocabTbody = document.getElementById("vocab-tbody");
  const vocabSummary = document.getElementById("vocab-summary");
  const sampleButtons = document.querySelectorAll(".sample-btn");

  const SUFFIXES = ["tion", "ment", "ness", "ing", "ed", "ly", "es", "s"];

  function hashTokenId(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return 100 + (Math.abs(h) % 899);
  }

  function tokenizeText(text) {
    const rawParts = text.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9']/g) || [];
    const tokens = [];

    for (const part of rawParts) {
      if (/^[^\w']+$/.test(part)) {
        tokens.push({ text: part, type: "Punctuation", id: hashTokenId(part) });
        continue;
      }

      // Split common plural / inflected suffixes for realistic subword feel
      let matchedSuffix = null;
      if (part.length > 4) {
        for (const suf of SUFFIXES) {
          if (part.toLowerCase().endsWith(suf) && part.length - suf.length >= 3) {
            matchedSuffix = part.slice(part.length - suf.length);
            const stem = part.slice(0, part.length - suf.length);
            tokens.push({ text: stem, type: "Word", id: hashTokenId(stem.toLowerCase()) });
            tokens.push({
              text: "▁" + matchedSuffix,
              raw: matchedSuffix,
              type: "Subword",
              id: hashTokenId("_" + matchedSuffix.toLowerCase()),
            });
            break;
          }
        }
      }

      if (!matchedSuffix) {
        tokens.push({ text: part, type: "Word", id: hashTokenId(part.toLowerCase()) });
      }
    }
    return tokens;
  }

  let currentTokens = [];

  function renderTokens() {
    const text = tokenInput.value.trim();
    currentTokens = tokenizeText(text);
    const byteLength = new TextEncoder().encode(text).length;

    tokenStatsBadge.textContent = `${currentTokens.length} tokens (${byteLength} bytes)`;
    tokensGrid.innerHTML = "";

    currentTokens.forEach((tok, idx) => {
      const palette = TOKEN_PALETTE[(tok.id + idx) % TOKEN_PALETTE.length];
      const chip = document.createElement("div");
      chip.className = "token-chip";
      chip.style.background = palette.bg;
      chip.style.borderColor = palette.border;
      chip.style.color = palette.text;

      const textSpan = document.createElement("span");
      textSpan.className = "token-text";
      textSpan.textContent = tok.text;

      const idSpan = document.createElement("span");
      idSpan.className = "token-id";
      idSpan.textContent = `#${tok.id}`;

      chip.appendChild(textSpan);
      chip.appendChild(idSpan);

      chip.addEventListener("mouseenter", (e) => showTooltip(e, tok, idx));
      chip.addEventListener("mousemove", moveTooltip);
      chip.addEventListener("mouseleave", hideTooltip);

      tokensGrid.appendChild(chip);
    });

    if (!vocabDrawer.hidden) {
      renderVocabulary();
    }

    if (typeof renderPositionalEncoding === "function") {
      renderPositionalEncoding();
    }
    if (typeof renderAttentionAndSoftmax === "function") {
      renderAttentionAndSoftmax();
    }
  }

  function showTooltip(e, tok, idx) {
    const bytes = new TextEncoder().encode(tok.raw || tok.text).length;
    tokenTooltip.innerHTML = `
      <div class="tooltip-row"><span class="tooltip-key">Token</span><span>"${tok.text}"</span></div>
      <div class="tooltip-row"><span class="tooltip-key">ID</span><span>#${tok.id}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Position</span><span>${idx}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Type</span><span>${tok.type}</span></div>
      <div class="tooltip-row"><span class="tooltip-key">Bytes</span><span>${bytes} B</span></div>
    `;
    tokenTooltip.hidden = false;
    moveTooltip(e);
  }

  function moveTooltip(e) {
    const pad = 14;
    tokenTooltip.style.left = `${Math.min(e.clientX + pad, window.innerWidth - 200)}px`;
    tokenTooltip.style.top = `${Math.min(e.clientY + pad, window.innerHeight - 130)}px`;
  }

  function hideTooltip() {
    tokenTooltip.hidden = true;
  }

  function renderVocabulary() {
    const map = new Map();
    for (const tok of currentTokens) {
      const key = `${tok.id}:${tok.text}`;
      if (!map.has(key)) {
        map.set(key, {
          id: tok.id,
          text: tok.text,
          type: tok.type,
          bytes: new TextEncoder().encode(tok.raw || tok.text).length,
          count: 0,
        });
      }
      map.get(key).count += 1;
    }

    const entries = Array.from(map.values()).sort((a, b) => a.id - b.id);
    vocabSummary.textContent = `${entries.length} unique tokens indexed from ${currentTokens.length} total tokens`;
    vocabTbody.innerHTML = entries
      .map(
        (entry) => `
      <tr>
        <td>#${entry.id}</td>
        <td>"${entry.text}"</td>
        <td>${entry.type}</td>
        <td>${entry.bytes} B</td>
        <td>${entry.count}</td>
      </tr>`
      )
      .join("");
  }

  sampleButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sampleButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      tokenInput.value = SAMPLE_SENTENCES[btn.dataset.sample];
      renderTokens();
    });
  });

  document.getElementById("btn-tokenize").addEventListener("click", renderTokens);
  tokenInput.addEventListener("input", renderTokens);

  document.getElementById("btn-vocab").addEventListener("click", () => {
    vocabDrawer.hidden = !vocabDrawer.hidden;
    if (!vocabDrawer.hidden) renderVocabulary();
  });

  document.getElementById("btn-close-vocab").addEventListener("click", () => {
    vocabDrawer.hidden = true;
  });

  // --------------------------------------------------------------------------
  // 4. Semantic Vectorization for Custom Text (Deterministic Keyword + Hash)
  // --------------------------------------------------------------------------
  const TOPIC_KEYWORDS = {
    ai: ["ai", "neural", "network", "llm", "model", "token", "transformer", "attention", "deep", "learn", "data", "code", "compute", "algorithm"],
    space: ["space", "star", "planet", "galaxy", "galaxies", "cosmos", "cosmic", "telescope", "nebula", "orbit", "astro", "moon", "sun", "universe"],
    food: ["pasta", "food", "cook", "chef", "bake", "bread", "sauce", "tomato", "basil", "recipe", "dish", "eat", "flavor", "kitchen"],
    marine: ["ocean", "sea", "whale", "dolphin", "coral", "turtle", "fish", "marine", "water", "swim", "reef", "echolocation", "deep"],
  };

  function computeCustomVector(text) {
    const words = (text.toLowerCase().match(/[a-z]+/g) || []);
    const base = [0, 0, 0];
    const centroids = {
      ai: [0.82, -0.45, 0.56],
      space: [-0.73, 0.67, 0.43],
      food: [0.22, 0.78, -0.58],
      marine: [-0.33, -0.78, -0.54],
    };

    let hits = 0;
    for (const w of words) {
      for (const [cluster, kwList] of Object.entries(TOPIC_KEYWORDS)) {
        if (kwList.some((kw) => w.includes(kw))) {
          base[0] += centroids[cluster][0];
          base[1] += centroids[cluster][1];
          base[2] += centroids[cluster][2];
          hits++;
        }
      }
    }

    // Add deterministic character-hash component so distinct sentences separate cleanly
    let h1 = 17, h2 = 31, h3 = 53;
    for (let i = 0; i < text.length; i++) {
      const c = text.charCodeAt(i);
      h1 = (h1 * 33 + c) % 1000;
      h2 = (h2 * 37 + c) % 1000;
      h3 = (h3 * 41 + c) % 1000;
    }

    const jitter = [
      ((h1 / 1000) - 0.5) * 0.45,
      ((h2 / 1000) - 0.5) * 0.45,
      ((h3 / 1000) - 0.5) * 0.45,
    ];

    if (hits === 0) {
      return [
        Math.max(-0.92, Math.min(0.92, jitter[0] * 1.8)),
        Math.max(-0.92, Math.min(0.92, jitter[1] * 1.8)),
        Math.max(-0.92, Math.min(0.92, jitter[2] * 1.8)),
      ];
    }

    return [
      Math.max(-0.95, Math.min(0.95, base[0] / hits + jitter[0] * 0.4)),
      Math.max(-0.95, Math.min(0.95, base[1] / hits + jitter[1] * 0.4)),
      Math.max(-0.95, Math.min(0.95, base[2] / hits + jitter[2] * 0.4)),
    ];
  }

  function cosineSimilarity(a, b) {
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const magA = Math.hypot(a[0], a[1], a[2]) || 1;
    const magB = Math.hypot(b[0], b[1], b[2]) || 1;
    return dot / (magA * magB);
  }

  // --------------------------------------------------------------------------
  // 5. Step 3: Sentence Comparison State & Sync
  // --------------------------------------------------------------------------
  const selectA = document.getElementById("select-a");
  const selectB = document.getElementById("select-b");
  const customA = document.getElementById("custom-a");
  const customB = document.getElementById("custom-b");
  const simScoreEl = document.getElementById("sim-score");
  const simBarFill = document.getElementById("sim-bar-fill");
  const simStatusDot = document.getElementById("sim-status-dot");
  const simStatusLabel = document.getElementById("sim-status-label");
  const simTextA = document.getElementById("sim-text-a");
  const simTextB = document.getElementById("sim-text-b");
  const simExplanation = document.getElementById("sim-explanation");

  let sentenceA = { ...DATASET_SENTENCES[1] }; // Deep neural networks...
  let sentenceB = { ...DATASET_SENTENCES[2] }; // Transformers use multi-head...

  function populateSelects() {
    const optionsHtml = DATASET_SENTENCES.map(
      (item) => `<option value="${item.id}">[${CLUSTERS[item.cluster].tag}] ${item.text}</option>`
    ).join("");
    selectA.innerHTML = optionsHtml;
    selectB.innerHTML = optionsHtml;
    selectA.value = sentenceA.id;
    selectB.value = sentenceB.id;
  }

  function updateComparison() {
    const rawCos = cosineSimilarity(sentenceA.vec, sentenceB.vec);
    // Map [-1, 1] -> [0%, 100%] or clamp positive semantic scale cleanly
    const pct = Math.max(0, Math.min(100, ((rawCos + 1) / 2) * 100));
    const displayPct = pct.toFixed(1);

    simScoreEl.textContent = `${displayPct}%`;
    simBarFill.style.width = `${displayPct}%`;
    simTextA.textContent = `"${sentenceA.text}"`;
    simTextB.textContent = `"${sentenceB.text}"`;

    if (pct >= 80) {
      simStatusDot.style.background = "#22c55e";
      simStatusDot.style.boxShadow = "0 0 10px #22c55e";
      simStatusLabel.textContent = "Very High Similarity (Same Conceptual Cluster)";
      simExplanation.textContent =
        "These sentences share primary topic anchors, high contextual overlap, and lie tightly clustered in 3D embedding space.";
    } else if (pct >= 55) {
      simStatusDot.style.background = "#eab308";
      simStatusDot.style.boxShadow = "0 0 10px #eab308";
      simStatusLabel.textContent = "Moderate Similarity (Adjacent Semantic Regions)";
      simExplanation.textContent =
        "These sentences share partial structural or secondary conceptual cues while occupying distinct regions of the embedding cube.";
    } else {
      simStatusDot.style.background = "#f43f5e";
      simStatusDot.style.boxShadow = "0 0 10px #f43f5e";
      simStatusLabel.textContent = "Low Similarity (Distant Conceptual Clusters)";
      simExplanation.textContent =
        "These sentences belong to unrelated domains and point in nearly opposite directions in 3D vector space.";
    }
  }

  selectA.addEventListener("change", () => {
    customA.value = "";
    const found = DATASET_SENTENCES.find((s) => s.id === selectA.value);
    if (found) sentenceA = { ...found };
    updateComparison();
  });

  selectB.addEventListener("change", () => {
    customB.value = "";
    const found = DATASET_SENTENCES.find((s) => s.id === selectB.value);
    if (found) sentenceB = { ...found };
    updateComparison();
  });

  customA.addEventListener("input", () => {
    const val = customA.value.trim();
    if (val) {
      sentenceA = { id: "custom-a", cluster: "custom", text: val, vec: computeCustomVector(val) };
    } else {
      const found = DATASET_SENTENCES.find((s) => s.id === selectA.value);
      if (found) sentenceA = { ...found };
    }
    updateComparison();
  });

  customB.addEventListener("input", () => {
    const val = customB.value.trim();
    if (val) {
      sentenceB = { id: "custom-b", cluster: "custom", text: val, vec: computeCustomVector(val) };
    } else {
      const found = DATASET_SENTENCES.find((s) => s.id === selectB.value);
      if (found) sentenceB = { ...found };
    }
    updateComparison();
  });

  document.getElementById("btn-project-3d").addEventListener("click", () => {
    const text = tokenInput.value.trim();
    if (!text) return;
    customA.value = text;
    sentenceA = { id: "custom-a", cluster: "custom", text, vec: computeCustomVector(text) };
    updateComparison();
    document.getElementById("step-2").scrollIntoView({ behavior: "smooth" });
  });

  // --------------------------------------------------------------------------
  // 6. Step 2: Interactive 3D Embedding Canvas Engine
  // --------------------------------------------------------------------------
  const canvas = document.getElementById("embedding-canvas");
  const ctx = canvas.getContext("2d");
  const nodeInspector = document.getElementById("node-inspector");
  const inspectorCat = document.getElementById("inspector-cat");
  const inspectorText = document.getElementById("inspector-text");
  const inspectorCoords = document.getElementById("inspector-coords");
  const btnAutoRotate = document.getElementById("btn-auto-rotate");

  let width = 800;
  let height = 490;
  let rotX = 0.35;
  let rotY = -0.55;
  let zoom = 1.05;
  let autoRotate = false;
  let dragging = false;
  let dragMoved = false;
  let lastMouse = { x: 0, y: 0 };
  let hoveredCluster = null;
  let selectedNode = null;
  let projectedNodes = [];

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    width = rect.width;
    height = rect.height;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function project3D(vec) {
    const [x, y, z] = vec;
    // Rotate around Y
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Rotate around X
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    const fov = 3.2;
    const scale = (Math.min(width, height) * 0.38 * zoom) / (fov + z2);
    return {
      x: width / 2 + x1 * scale,
      y: height / 2 - y2 * scale,
      depth: z2,
      scale,
    };
  }

  function drawLine3D(v1, v2, color, lineWidth = 1, dashed = false) {
    const p1 = project3D(v1);
    const p2 = project3D(v2);
    ctx.save();
    ctx.beginPath();
    if (dashed) ctx.setLineDash([6, 5]);
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.restore();
    return { p1, p2 };
  }

  function getActiveNodes() {
    const nodes = DATASET_SENTENCES.map((s) => ({ ...s }));
    if (sentenceA.id === "custom-a") nodes.push(sentenceA);
    if (sentenceB.id === "custom-b") nodes.push(sentenceB);
    return nodes;
  }

  function render3DScene() {
    if (autoRotate && !dragging) {
      rotY += 0.0032;
    }

    ctx.clearRect(0, 0, width, height);

    // 1. Draw 3D Bounding Cube [-1, 1]^3
    const cubeVerts = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [i, j] of edges) {
      drawLine3D(cubeVerts[i], cubeVerts[j], "rgba(148, 163, 184, 0.14)", 1);
    }

    // 2. Origin Axes
    drawLine3D([0, 0, 0], [0.42, 0, 0], "rgba(56, 189, 248, 0.45)", 1.5);
    drawLine3D([0, 0, 0], [0, 0.42, 0], "rgba(168, 85, 247, 0.45)", 1.5);
    drawLine3D([0, 0, 0], [0, 0, 0.42], "rgba(16, 185, 129, 0.45)", 1.5);

    // 3. Dashed Similarity Vector between Sentence A & Sentence B
    const { p1, p2 } = drawLine3D(sentenceA.vec, sentenceB.vec, "rgba(255, 255, 255, 0.72)", 2, true);
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;
    const dist = Math.hypot(
      sentenceA.vec[0] - sentenceB.vec[0],
      sentenceA.vec[1] - sentenceB.vec[1],
      sentenceA.vec[2] - sentenceB.vec[2]
    ).toFixed(2);

    ctx.save();
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    const distLabel = `DIST ${dist}`;
    const tw = ctx.measureText(distLabel).width + 12;
    ctx.fillStyle = "rgba(6, 9, 19, 0.88)";
    ctx.strokeStyle = "rgba(34, 211, 238, 0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(midX - tw / 2, midY - 10, tw, 20, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#e2e8f0";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(distLabel, midX, midY);
    ctx.restore();

    // 4. Project & Depth-sort Nodes
    const activeNodes = getActiveNodes();
    projectedNodes = activeNodes
      .map((node) => ({
        node,
        proj: project3D(node.vec),
      }))
      .sort((a, b) => b.proj.depth - a.proj.depth);

    for (const { node, proj } of projectedNodes) {
      const isA = node.id === sentenceA.id || (node.text === sentenceA.text && node.id === sentenceA.id);
      const isB = node.id === sentenceB.id || (node.text === sentenceB.text && node.id === sentenceB.id);
      const isSelected = selectedNode && selectedNode.id === node.id;
      const dimmed = hoveredCluster && hoveredCluster !== node.cluster && !isA && !isB;
      const color = CLUSTERS[node.cluster].color;
      const radius = isA || isB || isSelected ? 7.5 : 5.5;

      ctx.save();
      ctx.globalAlpha = dimmed ? 0.22 : 1;

      // Glow
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, radius * 2.1, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = dimmed ? 0.05 : 0.22;
      ctx.fill();

      // Core node
      ctx.globalAlpha = dimmed ? 0.25 : 1;
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = isA || isB || isSelected ? 2.2 : 1.2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();

      // Short label
      const shortText = node.text.length > 28 ? node.text.slice(0, 27) + "…" : node.text;
      ctx.font = "500 11px 'Plus Jakarta Sans', sans-serif";
      ctx.fillStyle = "rgba(241, 245, 249, 0.88)";
      ctx.textBaseline = "middle";
      ctx.fillText(shortText, proj.x + radius + 6, proj.y);

      if (isA || isB) {
        const tag = isA ? "SENTENCE A" : "SENTENCE B";
        ctx.font = "700 9px 'JetBrains Mono', monospace";
        ctx.fillStyle = isA ? "#38bdf8" : "#f472b6";
        ctx.fillText(tag, proj.x + radius + 6, proj.y - 12);
      }

      ctx.restore();
    }

    requestAnimationFrame(render3DScene);
  }

  // Canvas Pointer / Mouse Interactions (supports mouse, trackpad, & touch)
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    dragging = true;
    dragMoved = false;
    lastMouse = { x: e.clientX, y: e.clientY };
    if (canvas.setPointerCapture) {
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch (_) {}
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) dragMoved = true;
    rotY += dx * 0.016;
    rotX = Math.max(-1.35, Math.min(1.35, rotX + dy * 0.016));
    lastMouse = { x: e.clientX, y: e.clientY };
  });

  const stopDragging = (e) => {
    dragging = false;
    if (e && canvas.releasePointerCapture) {
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (_) {}
    }
  };

  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  window.addEventListener("blur", () => {
    dragging = false;
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      zoom = Math.max(0.55, Math.min(2.1, zoom - e.deltaY * 0.001));
    },
    { passive: false }
  );

  canvas.addEventListener("click", (e) => {
    if (dragMoved) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Find closest node within 16px (front-most first)
    let hit = null;
    for (let i = projectedNodes.length - 1; i >= 0; i--) {
      const item = projectedNodes[i];
      if (Math.hypot(item.proj.x - mx, item.proj.y - my) < 16) {
        hit = item.node;
        break;
      }
    }

    if (hit) {
      selectedNode = hit;
      inspectorCat.textContent = CLUSTERS[hit.cluster].label;
      inspectorCat.style.color = CLUSTERS[hit.cluster].color;
      inspectorText.textContent = hit.text;
      inspectorCoords.textContent = `vec: [${hit.vec.map((n) => n.toFixed(2)).join(", ")}]`;
      nodeInspector.hidden = false;
    }
  });

  document.getElementById("btn-close-inspector").addEventListener("click", () => {
    nodeInspector.hidden = true;
    selectedNode = null;
  });

  document.getElementById("btn-set-a").addEventListener("click", () => {
    if (!selectedNode) return;
    sentenceA = { ...selectedNode };
    if (DATASET_SENTENCES.some((s) => s.id === selectedNode.id)) {
      selectA.value = selectedNode.id;
      customA.value = "";
    } else {
      customA.value = selectedNode.text;
    }
    updateComparison();
  });

  document.getElementById("btn-set-b").addEventListener("click", () => {
    if (!selectedNode) return;
    sentenceB = { ...selectedNode };
    if (DATASET_SENTENCES.some((s) => s.id === selectedNode.id)) {
      selectB.value = selectedNode.id;
      customB.value = "";
    } else {
      customB.value = selectedNode.text;
    }
    updateComparison();
  });

  btnAutoRotate.addEventListener("click", () => {
    autoRotate = !autoRotate;
    btnAutoRotate.classList.toggle("active", autoRotate);
    btnAutoRotate.textContent = `Auto-Orbit: ${autoRotate ? "On" : "Off"}`;
  });

  document.getElementById("btn-reset-view").addEventListener("click", () => {
    rotX = 0.35;
    rotY = -0.55;
    zoom = 1.05;
  });

  document.querySelectorAll(".legend-item").forEach((item) => {
    item.addEventListener("mouseenter", () => {
      hoveredCluster = item.dataset.cluster;
    });
    item.addEventListener("mouseleave", () => {
      hoveredCluster = null;
    });
  });

  window.addEventListener("resize", () => {
    resizeCanvas();
    renderPositionalEncoding();
  });

  // --------------------------------------------------------------------------
  // 7. Active Pipeline Stages Bar Sync
  // --------------------------------------------------------------------------
  const stageCards = document.querySelectorAll(".stage-card");
  stageCards.forEach((card) => {
    card.addEventListener("click", () => {
      stageCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
    });
  });

  // --------------------------------------------------------------------------
  // 8. Step 4: Sinusoidal Positional Encoding (Word Order & Waves)
  // --------------------------------------------------------------------------
  const peTokenStrip = document.getElementById("pe-token-strip");
  const peWaveCanvas = document.getElementById("pe-wave-canvas");
  const peWaveCtx = peWaveCanvas.getContext("2d");
  const peHeatmap = document.getElementById("pe-heatmap");
  const PE_DIMS = 16;
  let activePePos = 0;

  function peValue(pos, dim) {
    const angle = pos / Math.pow(10000, (2 * Math.floor(dim / 2)) / PE_DIMS);
    return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  }

  function renderPositionalEncoding() {
    const tokens = currentTokens.slice(0, 14);
    if (tokens.length === 0) return;
    if (activePePos >= tokens.length) activePePos = 0;

    // 1. Token selector pills
    peTokenStrip.innerHTML = tokens
      .map(
        (t, idx) =>
          `<button type="button" class="pe-tok-pill ${idx === activePePos ? "active" : ""}" data-pos="${idx}">[${idx}] ${t.text}</button>`
      )
      .join("");

    peTokenStrip.querySelectorAll(".pe-tok-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        activePePos = Number(pill.dataset.pos);
        renderPositionalEncoding();
      });
    });

    // 2. Sinusoidal Wave Canvas
    const rect = peWaveCanvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(280, rect.width - 34);
    const h = 210;
    peWaveCanvas.width = Math.floor(w * dpr);
    peWaveCanvas.height = Math.floor(h * dpr);
    peWaveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    peWaveCtx.clearRect(0, 0, w, h);

    // Midline
    peWaveCtx.strokeStyle = "rgba(148, 163, 184, 0.18)";
    peWaveCtx.lineWidth = 1;
    peWaveCtx.beginPath();
    peWaveCtx.moveTo(0, h / 2);
    peWaveCtx.lineTo(w, h / 2);
    peWaveCtx.stroke();

    const maxPos = Math.max(tokens.length - 1, 1);
    const curves = [
      { dim: 0, color: "#38bdf8" },
      { dim: 1, color: "#a855f7" },
      { dim: 4, color: "#10b981" },
      { dim: 8, color: "#fb923c" },
    ];

    for (const c of curves) {
      peWaveCtx.beginPath();
      peWaveCtx.strokeStyle = c.color;
      peWaveCtx.lineWidth = 2;
      const steps = 80;
      for (let s = 0; s <= steps; s++) {
        const pos = (s / steps) * maxPos;
        const x = 14 + (s / steps) * (w - 28);
        const y = h / 2 - peValue(pos, c.dim) * (h * 0.36);
        if (s === 0) peWaveCtx.moveTo(x, y);
        else peWaveCtx.lineTo(x, y);
      }
      peWaveCtx.stroke();
    }

    // Highlight active token position marker
    const markerX = 14 + (activePePos / maxPos) * (w - 28);
    peWaveCtx.save();
    peWaveCtx.setLineDash([4, 4]);
    peWaveCtx.strokeStyle = "rgba(255, 255, 255, 0.6)";
    peWaveCtx.beginPath();
    peWaveCtx.moveTo(markerX, 10);
    peWaveCtx.lineTo(markerX, h - 10);
    peWaveCtx.stroke();
    peWaveCtx.restore();

    // 3. Positional Heatmap
    peHeatmap.innerHTML = tokens
      .map((tok, pos) => {
        let cells = "";
        for (let d = 0; d < PE_DIMS; d++) {
          const val = peValue(pos, d); // [-1, 1]
          const bg =
            val >= 0
              ? `rgba(34, 211, 238, ${(0.15 + val * 0.85).toFixed(2)})`
              : `rgba(168, 85, 247, ${(0.15 + Math.abs(val) * 0.85).toFixed(2)})`;
          cells += `<div class="pe-heat-cell" style="background:${bg}" title="pos ${pos}, dim ${d}: ${val.toFixed(3)}"></div>`;
        }
        return `<div class="pe-heat-row ${pos === activePePos ? "active" : ""}">
          <span class="pe-heat-label">${pos}: ${tok.text}</span>${cells}
        </div>`;
      })
      .join("");
  }

  // --------------------------------------------------------------------------
  // 9. Step 5: Self-Attention & Softmax Next-Token Generation
  // --------------------------------------------------------------------------
  const attnTokensEl = document.getElementById("attn-tokens");
  const attnBarsEl = document.getElementById("attn-bars");
  const tempSlider = document.getElementById("temp-slider");
  const tempValEl = document.getElementById("temp-val");
  const softmaxListEl = document.getElementById("softmax-list");
  const btnAppendToken = document.getElementById("btn-append-token");

  let activeQueryIdx = 0;
  let currentTopPrediction = "embeddings";

  const NEXT_TOKEN_CANDIDATES = {
    ai: [
      { tok: "representations", logit: 4.8 },
      { tok: "patterns", logit: 4.2 },
      { tok: "probabilities", logit: 3.7 },
      { tok: "embeddings", logit: 3.1 },
      { tok: "sequences", logit: 2.5 },
    ],
    space: [
      { tok: "galaxies", logit: 4.7 },
      { tok: "nebulae", logit: 4.1 },
      { tok: "supernovae", logit: 3.4 },
      { tok: "exoplanets", logit: 2.9 },
      { tok: "quasars", logit: 2.2 },
    ],
    food: [
      { tok: "basil", logit: 4.9 },
      { tok: "parmesan", logit: 4.3 },
      { tok: "garlic", logit: 3.6 },
      { tok: "oregano", logit: 2.8 },
      { tok: "truffle", logit: 2.1 },
    ],
    marine: [
      { tok: "currents", logit: 4.6 },
      { tok: "habitats", logit: 4.1 },
      { tok: "ecosystems", logit: 3.5 },
      { tok: "depths", logit: 2.9 },
      { tok: "reefs", logit: 2.4 },
    ],
  };

  function softmax(logits, temp) {
    const scaled = logits.map((l) => l / Math.max(0.05, temp));
    const maxL = Math.max(...scaled);
    const exps = scaled.map((s) => Math.exp(s - maxL));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  function detectDominantTopic(text) {
    const lower = text.toLowerCase();
    let best = "ai";
    let bestHits = -1;
    for (const [topic, kws] of Object.entries(TOPIC_KEYWORDS)) {
      const hits = kws.filter((k) => lower.includes(k)).length;
      if (hits > bestHits) {
        bestHits = hits;
        best = topic;
      }
    }
    return best;
  }

  function renderAttentionAndSoftmax() {
    const tokens = currentTokens.slice(0, 14);
    if (tokens.length === 0) return;
    if (activeQueryIdx >= tokens.length) activeQueryIdx = tokens.length - 1;

    // 1. Query Token Buttons
    attnTokensEl.innerHTML = tokens
      .map(
        (t, idx) =>
          `<button type="button" class="attn-tok-btn ${idx === activeQueryIdx ? "active" : ""}" data-idx="${idx}">${t.text}</button>`
      )
      .join("");

    attnTokensEl.querySelectorAll(".attn-tok-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeQueryIdx = Number(btn.dataset.idx);
        renderAttentionAndSoftmax();
      });
    });

    // 2. Causal Self-Attention Weights (query attends to keys 0..activeQueryIdx)
    const rawScores = tokens.map((t, idx) => {
      if (idx > activeQueryIdx) return -1e9; // causal mask
      const dist = activeQueryIdx - idx;
      const sameLenBonus = Math.abs(t.text.length - tokens[activeQueryIdx].text.length) < 2 ? 0.6 : 0;
      const selfBonus = idx === activeQueryIdx ? 1.4 : 0;
      return 2.2 - dist * 0.28 + sameLenBonus + selfBonus + ((t.id % 7) * 0.12);
    });

    const weights = softmax(rawScores, 1.0);
    attnBarsEl.innerHTML = tokens
      .map((t, idx) => {
        const pct = (weights[idx] * 100).toFixed(1);
        const masked = idx > activeQueryIdx;
        return `<div class="attn-bar-row" style="opacity:${masked ? 0.28 : 1}">
          <span class="attn-bar-tok">${idx}: ${t.text}</span>
          <div class="attn-track"><div class="attn-fill" style="width:${pct}%"></div></div>
          <span class="attn-pct">${masked ? "masked" : pct + "%"}</span>
        </div>`;
      })
      .join("");

    // 3. Softmax Next-Token Probabilities
    const temp = parseFloat(tempSlider.value);
    tempValEl.textContent = temp.toFixed(1);
    const topic = detectDominantTopic(tokenInput.value);
    const candidates = NEXT_TOKEN_CANDIDATES[topic] || NEXT_TOKEN_CANDIDATES.ai;
    const probs = softmax(
      candidates.map((c) => c.logit),
      temp
    );
    currentTopPrediction = candidates[0].tok;

    softmaxListEl.innerHTML = candidates
      .map((c, idx) => {
        const pct = (probs[idx] * 100).toFixed(1);
        return `<div class="softmax-row">
          <span class="attn-bar-tok">"${c.tok}"</span>
          <div class="attn-track"><div class="attn-fill" style="width:${pct}%"></div></div>
          <span class="attn-pct">${pct}%</span>
        </div>`;
      })
      .join("");
  }

  tempSlider.addEventListener("input", renderAttentionAndSoftmax);

  btnAppendToken.addEventListener("click", () => {
    const cur = tokenInput.value.trim();
    const cleaned = cur.endsWith(".") ? cur.slice(0, -1) : cur;
    tokenInput.value = `${cleaned} ${currentTopPrediction}.`;
    renderTokens();
  });

  // --------------------------------------------------------------------------
  // 10. Initialize App
  // --------------------------------------------------------------------------
  renderTokens();
  populateSelects();
  updateComparison();
  resizeCanvas();
  requestAnimationFrame(render3DScene);
})();
