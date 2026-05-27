from __future__ import annotations

HTML_SIMULATOR_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  body {
    background-color: transparent;
    color: #1f1f1f;
    margin: 0;
    padding: 0;
    font-family: 'Inter', sans-serif;
    overflow: hidden;
    user-select: none;
  }
  
  .stage {
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 310px;
    background: #f0f4f9;
    border: 1px solid #c4c7c5;
    border-radius: 16px;
    padding: 10px 24px;
    position: relative;
    box-sizing: border-box;
  }
  
  .nodes-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 58%;
    height: 100%;
    position: relative;
    z-index: 2;
  }
  
  /* Material 3 Stepper Logs Container */
  .stepper-container {
    width: 39%;
    height: 286px;
    background: #ffffff;
    border: 1px solid #e1e2e9;
    border-radius: 12px;
    padding: 16px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    position: relative;
  }
  
  .stepper-title {
    font-size: 0.78rem;
    color: #0b57d0;
    border-bottom: 1px solid #e1e2e9;
    padding-bottom: 8px;
    margin-bottom: 12px;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  
  .stepper-steps {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    justify-content: center;
  }
  
  .step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.78rem;
    color: #5e5e5e;
    opacity: 0.25;
    transform: translateY(2px);
    transition: all 0.3s cubic-bezier(0.2, 0, 0, 1);
  }
  
  .step-row.active {
    opacity: 1;
    color: #1f1f1f;
    transform: translateY(0);
  }
  
  .step-icon-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #f0f4f9;
    color: #0b57d0;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    transition: all 0.2s ease;
  }
  
  .step-row.active .step-icon-wrap {
    background-color: #d3e3fd;
  }
  
  .step-row.success .step-icon-wrap {
    background-color: #e8f5e9;
    color: #146c2e;
  }
  
  .step-row.danger .step-icon-wrap {
    background-color: #f9dedc;
    color: #b3261e;
  }

  /* Node styling */
  .node {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    width: 90px;
    height: 140px;
    justify-content: center;
  }
  
  .node-icon-wrap {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #ffffff;
    border: 1px solid #c4c7c5;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #5e5e5e;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    transition: all 0.3s cubic-bezier(0.2, 0, 0, 1);
    z-index: 3;
    position: relative;
  }
  
  .node.client .node-icon-wrap {
    background-color: #e8f0fe;
    border-color: #aecbfa;
    color: #0b57d0;
  }
  
  /* State animations for Google Material 3 */
  .state-normal .node.firewall .node-icon-wrap {
    background-color: #e8f5e9;
    border-color: #a3cfbb;
    color: #146c2e;
    transform: scale(1.05);
  }
  
  .state-normal .node.server .node-icon-wrap {
    background-color: #e8f5e9;
    border-color: #a3cfbb;
    color: #146c2e;
  }
  
  .state-blocked .node.firewall .node-icon-wrap {
    background-color: #f9dedc;
    border-color: #f5c2c7;
    color: #b3261e;
    transform: scale(1.05);
    animation: srv-shake 0.15s 2;
  }
  
  @keyframes srv-shake {
    0%, 100% { transform: scale(1.05) translateX(0); }
    50% { transform: scale(1.05) translateX(-2px); }
  }

  .node-label {
    position: absolute;
    bottom: 5px;
    left: 0;
    right: 0;
    font-size: 0.68rem;
    font-weight: 600;
    color: #5e5e5e;
    text-align: center;
    letter-spacing: 0.02em;
  }
  
  .state-normal .node.client .node-label { color: #0b57d0; }
  .state-normal .node.firewall .node-label { color: #146c2e; }
  .state-normal .node.server .node-label { color: #146c2e; }
  
  .state-blocked .node.client .node-label { color: #b3261e; }
  .state-blocked .node.firewall .node-label { color: #b3261e; }

  /* Connecting Rails */
  .connectors-svg {
    position: absolute;
    top: 0;
    left: 45px;
    width: calc(58% - 90px);
    height: 100%;
    z-index: 1;
    pointer-events: none;
  }
  
  /* Packets visual */
  .packet {
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #0b57d0;
    top: 50%;
    margin-top: -7px; /* aligns packet center exactly to connection rails */
    z-index: 2;
    opacity: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    justify-content: center;
    align-items: center;
    color: #ffffff;
    font-size: 0.48rem;
    font-weight: bold;
  }
  
  /* Ripple wave for block animation */
  .ripple-shield {
    position: absolute;
    border-radius: 50%;
    border: 2px solid #b3261e;
    background: rgba(179, 38, 30, 0.05);
    opacity: 0;
    pointer-events: none;
    z-index: 4;
    transform: translate(-50%, -50%);
  }
</style>
</head>
<body class="state-idle">
<div class="stage" id="stage">
  <svg class="connectors-svg">
    <!-- Clean Google network track line -->
    <line x1="0%" y1="50%" x2="100%" y2="50%" stroke="#c4c7c5" stroke-width="3" id="track" />
  </svg>
  
  <div class="nodes-container">
    <div class="node client">
      <div class="node-icon-wrap"><i data-lucide="monitor" style="width: 24px; height: 24px;"></i></div>
      <div class="node-label" id="client-label">Source</div>
    </div>
    
    <div class="node firewall">
      <div class="node-icon-wrap" id="fw-icon"><i data-lucide="shield" style="width: 24px; height: 24px;"></i></div>
      <div class="node-label" id="fw-label">IDS Gateway</div>
    </div>
    
    <div class="node server">
      <div class="node-icon-wrap" id="srv-icon"><i data-lucide="server" style="width: 24px; height: 24px;"></i></div>
      <div class="node-label">Serveur</div>
    </div>
  </div>
  
  <div class="stepper-container">
    <div class="stepper-title">Statut d'Analyse</div>
    <div class="stepper-steps" id="stepper-steps">
      <div class="step-row" id="step-1">
        <div class="step-icon-wrap">1</div>
        <span>Détection du flux réseau...</span>
      </div>
      <div class="step-row" id="step-2">
        <div class="step-icon-wrap">2</div>
        <span>Extraction des métadonnées...</span>
      </div>
      <div class="step-row" id="step-3">
        <div class="step-icon-wrap">3</div>
        <span>Évaluation LightGBM...</span>
      </div>
      <div class="step-row" id="step-4">
        <div class="step-icon-wrap">4</div>
        <span>Décision finale de sécurité</span>
      </div>
    </div>
  </div>
</div>

<script>
  // Dynamic run token to force iframe reloads on button re-clicks
  const runToken = "__RUN_TOKEN__";
  const status = "__STATUS__"; // 'idle' | 'normal' | 'blocked'
  const label = "__LABEL__";
  const score = __SCORE__;
  const threshold = __THRESHOLD__;
  const category = "__CATEGORY__";
  
  const protocol = "__PROTOCOL__";
  const service = "__SERVICE__";
  const srcBytes = "__SRC_BYTES__";
  const count = "__COUNT__";
  const scenario = "__SCENARIO__";

  const stageEl = document.body;
  const nodesContainer = document.querySelector('.nodes-container');

  // Activate icons
  lucide.createIcons();

  function setStep(stepNum, state, text = null) {
    const el = document.getElementById(`step-${stepNum}`);
    if (!el) return;
    
    el.className = 'step-row active';
    if (state === 'success') {
      el.classList.add('success');
      el.querySelector('.step-icon-wrap').innerHTML = '<i data-lucide="check" style="width: 12px; height: 12px;"></i>';
    } else if (state === 'danger') {
      el.classList.add('danger');
      el.querySelector('.step-icon-wrap').innerHTML = '<i data-lucide="x" style="width: 12px; height: 12px;"></i>';
    }
    
    if (text) {
      el.querySelector('span').innerText = text;
    }
    lucide.createIcons();
  }

  function resetSimulationState() {
    stageEl.className = 'state-idle';
    document.getElementById('client-label').innerText = 'Source';
    document.getElementById('fw-label').innerText = 'IDS Gateway';
    const fwIcon = document.getElementById('fw-icon');
    fwIcon.innerHTML = '<i data-lucide="shield" style="width: 24px; height: 24px;"></i>';
    fwIcon.style.animation = 'none';
    fwIcon.style.transform = 'scale(1)';
    fwIcon.style.borderColor = '#c4c7c5';
    
    const srvIcon = document.getElementById('srv-icon');
    srvIcon.innerHTML = '<i data-lucide="server" style="width: 24px; height: 24px;"></i>';
    srvIcon.style.transform = 'scale(1)';
    srvIcon.style.borderColor = '#c4c7c5';
    
    document.querySelectorAll('.packet, .ripple-shield').forEach(el => el.remove());
    
    const stepTexts = {
      1: "Détection du flux réseau...",
      2: "Extraction des métadonnées...",
      3: "Évaluation LightGBM...",
      4: "Décision finale de sécurité"
    };
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      if (el) {
        el.className = 'step-row';
        el.querySelector('span').innerText = stepTexts[i];
        el.querySelector('.step-icon-wrap').innerHTML = i;
      }
    }
    lucide.createIcons();
  }

  function startSimulation() {
    resetSimulationState();

    if (status === 'idle') {
      setStep(1, 'active', 'Attente d\\'une connexion...');
      return;
    }

    // Step 1: Detect
    if (scenario === 'ddos_syn') {
      setStep(1, 'success', 'Flux TCP/SYN Détecté');
      setTimeout(() => setStep(2, 'success', 'Extraction: SYN Flood suspect (480 reqs)'), 300);
    } else if (scenario === 'teardrop') {
      setStep(1, 'success', 'Flux UDP/IP Détecté');
      setTimeout(() => setStep(2, 'success', 'Extraction: Offsets superposés invalides'), 300);
    } else if (scenario === 'ping_death') {
      setStep(1, 'success', 'Flux ICMP/Echo Détecté');
      setTimeout(() => setStep(2, 'success', 'Extraction: Paquet ICMP géant (65510B)'), 300);
    } else if (scenario === 'nmap_scan') {
      setStep(1, 'success', 'Balayage TCP Détecté');
      setTimeout(() => setStep(2, 'success', 'Extraction: Scan furtif vertical (Nmap)'), 300);
    } else if (scenario === 'sql_injection') {
      setStep(1, 'success', 'Requête HTTP POST Détectée');
      setTimeout(() => setStep(2, 'success', 'Extraction: Requête SQL applicative'), 300);
    } else if (scenario === 'buffer_overflow') {
      setStep(1, 'success', 'Connexion TCP/Telnet Détectée');
      setTimeout(() => setStep(2, 'success', 'Extraction: Shellcode d\\'élévation privilèges'), 300);
    } else if (scenario === 'ssh_bruteforce') {
      setStep(1, 'success', 'Connexions SSH Répétées');
      setTimeout(() => setStep(2, 'success', 'Extraction: 12 échecs d\\'authentification'), 300);
    } else if (scenario === 'backdoor') {
      setStep(1, 'success', 'Flux TCP/Backdoor Détecté');
      setTimeout(() => setStep(2, 'success', 'Extraction: Signal de shell distant entrant'), 300);
    } else {
      setStep(1, 'success', `Flux TCP/${protocol.toUpperCase()} détecté`);
      setTimeout(() => setStep(2, 'success', `Données extraites (${srcBytes} octets)`), 300);
    }

    // Step 2 activates animations at 400ms
    setTimeout(() => {
      animatePackets();
    }, 400);
  }

  function animatePackets() {
    const nodesRect = nodesContainer.getBoundingClientRect();
    const startX = 45;
    const fwX = nodesRect.width / 2;
    const endX = nodesRect.width - 45;
    const centerY = nodesRect.height * 0.5;

    if (scenario === 'ddos_syn') {
      // DDoS: dense burst of small red packets, staggered across the lane.
      let spawned = 0;
      const interval = setInterval(() => {
        spawnSinglePacket(startX, fwX, endX, centerY, 'ddos_syn', spawned);
        spawned++;
        if (spawned >= 90) {
          clearInterval(interval);
        }
      }, 12);
    } else if (scenario === 'teardrop') {
      // 3 overlapping orange ovals
      for (let i = 0; i < 3; i++) {
        setTimeout(() => {
          spawnSinglePacket(startX, fwX, endX, centerY, 'teardrop', i);
        }, i * 60);
      }
    } else if (scenario === 'nmap_scan') {
      // 8 rapid yellow scan points
      for (let i = 0; i < 8; i++) {
        setTimeout(() => {
          spawnSinglePacket(startX, fwX, endX, centerY, 'nmap_scan', i);
        }, i * 110);
      }
    } else if (scenario === 'ssh_bruteforce') {
      // 8 rapid ssh brute force keys/lock packets
      for (let i = 0; i < 8; i++) {
        setTimeout(() => {
          spawnSinglePacket(startX, fwX, endX, centerY, 'ssh_bruteforce', i);
        }, i * 100);
      }
    } else if (scenario === 'backdoor') {
      spawnSinglePacket(startX, fwX, endX, centerY, 'backdoor', 0);
    } else {
      // Standard normal or other single packet attacks
      spawnSinglePacket(startX, fwX, endX, centerY, scenario, 0);
    }
  }

  function spawnSinglePacket(startX, fwX, endX, yPos, type, index) {
    const p = document.createElement('div');
    p.className = 'packet';
    
    // Set custom packet vertical alignment depending on type
    let finalY = yPos;
    
    if (type === 'ddos_syn') {
      const size = 6 + Math.floor(Math.random() * 5);
      p.style.width = size + 'px';
      p.style.height = size + 'px';
      p.style.marginTop = -(size / 2) + 'px';
      p.style.backgroundColor = '#dc2626';
      p.style.boxShadow = '0 0 8px rgba(220,38,38,0.65)';
      finalY = yPos + (Math.random() - 0.5) * 72;
      p.innerHTML = '';
    } else if (type === 'teardrop') {
      p.style.width = '30px';
      p.style.height = '12px';
      p.style.borderRadius = '4px';
      p.style.marginTop = '-6px';
      p.style.backgroundColor = '#d97706';
      p.style.boxShadow = '0 0 4px #d97706';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-20px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#fff3e0';
      badge.style.border = '1px solid #ffb74d';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 3px';
      badge.style.fontSize = '0.5rem';
      badge.style.color = '#d97706';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      
      if (index === 0) badge.innerText = "offset=0 (len=36)";
      else if (index === 1) badge.innerText = "offset=20 (len=28, overlap!)";
      else badge.innerText = "offset=12 (len=32, overlap!)";
      
      p.appendChild(badge);
    } else if (type === 'ping_death') {
      p.style.width = '36px';
      p.style.height = '36px';
      p.style.marginTop = '-18px';
      p.style.backgroundColor = '#b3261e';
      p.style.boxShadow = '0 0 12px rgba(179,38,30,0.6)';
      p.innerHTML = '<span style="font-size: 0.65rem; font-weight: bold; color: white; line-height: 36px;">65K</span>';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-20px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#f9dedc';
      badge.style.border = '1px solid #f5c2c7';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 3px';
      badge.style.fontSize = '0.5rem';
      badge.style.color = '#b3261e';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      badge.innerText = "Payload: 65,510 bytes (Malformed)";
      p.appendChild(badge);
    } else if (type === 'nmap_scan') {
      p.style.width = '10px';
      p.style.height = '10px';
      p.style.marginTop = '-5px';
      p.style.backgroundColor = '#eab308';
      p.style.boxShadow = '0 0 6px #eab308';
      finalY = yPos + (index % 3 - 1) * 12;
      p.innerHTML = '';
    } else if (type === 'sql_injection') {
      p.style.width = '16px';
      p.style.height = '16px';
      p.style.marginTop = '-8px';
      p.style.backgroundColor = '#7c3aed';
      p.style.boxShadow = '0 0 8px #7c3aed';
      p.innerHTML = '<i data-lucide="database" style="width: 10px; height: 10px; color: white; margin-top: 3px;"></i>';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-22px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#f3e8ff';
      badge.style.border = '1px solid #c084fc';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 4px';
      badge.style.fontSize = '0.55rem';
      badge.style.color = '#7c3aed';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      badge.innerText = index === 0 ? "admin' OR 1=1 --" : "UNION SELECT password...";
      p.appendChild(badge);
    } else if (type === 'buffer_overflow') {
      p.style.width = '16px';
      p.style.height = '16px';
      p.style.marginTop = '-8px';
      p.style.backgroundColor = '#7c3aed';
      p.style.boxShadow = '0 0 8px #7c3aed';
      p.innerHTML = '<i data-lucide="terminal" style="width: 10px; height: 10px; color: white; margin-top: 3px;"></i>';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-22px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#f3e8ff';
      badge.style.border = '1px solid #c084fc';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 4px';
      badge.style.fontSize = '0.55rem';
      badge.style.color = '#7c3aed';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      badge.innerText = category === 'u2r' ? "\\x90\\x90\\x90\\x90 (NOP Slide)" : "root# (Shellcode)";
      p.appendChild(badge);
    } else if (type === 'ssh_bruteforce') {
      p.style.width = '12px';
      p.style.height = '12px';
      p.style.marginTop = '-6px';
      p.style.backgroundColor = '#dc2626';
      p.style.boxShadow = '0 0 6px #dc2626';
      p.innerHTML = '<i data-lucide="key" style="width: 8px; height: 8px; color: white; margin-top: 2px;"></i>';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-20px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#f9dedc';
      badge.style.border = '1px solid #f5c2c7';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 3px';
      badge.style.fontSize = '0.5rem';
      badge.style.color = '#b3261e';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      
      const logins = ["root / admin", "admin / admin", "user / password", "root / 123456", "mysql / mysql", "oracle / system", "root / root", "admin / 1234"];
      badge.innerText = logins[index % logins.length] + " (Echec)";
      p.appendChild(badge);
    } else if (type === 'backdoor_reverse') {
      p.style.width = '16px';
      p.style.height = '16px';
      p.style.marginTop = '-8px';
      p.style.backgroundColor = '#d017a0';
      p.style.boxShadow = '0 0 10px #d017a0';
      p.innerHTML = '<i data-lucide="terminal" style="width: 10px; height: 10px; color: white; margin-top: 3px;"></i>';
      
      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      badge.style.top = '-22px';
      badge.style.left = '50%';
      badge.style.transform = 'translateX(-50%)';
      badge.style.background = '#fce7f3';
      badge.style.border = '1px solid #fbcfe8';
      badge.style.borderRadius = '3px';
      badge.style.padding = '1px 4px';
      badge.style.fontSize = '0.55rem';
      badge.style.color = '#be185d';
      badge.style.fontFamily = 'monospace';
      badge.style.whiteSpace = 'nowrap';
      badge.innerText = "Reverse Shell outbound: /bin/bash -> attacker:4444";
      p.appendChild(badge);
    } else if (type === 'backdoor') {
      p.style.width = '16px';
      p.style.height = '16px';
      p.style.marginTop = '-8px';
      p.style.backgroundColor = '#d017a0';
      p.style.boxShadow = '0 0 10px rgba(208,23,160,0.55)';
      p.innerHTML = '<i data-lucide="terminal" style="width: 10px; height: 10px; color: white; margin-top: 3px;"></i>';
    } else {
      p.style.backgroundColor = '#0b57d0';
      p.innerHTML = '<i data-lucide="activity" style="width: 8px; height: 8px; color: white; margin-top: 3px;"></i>';
    }

    p.style.left = (startX - (p.offsetWidth || 14)/2) + 'px';
    p.style.top = finalY + 'px';
    nodesContainer.appendChild(p);
    lucide.createIcons();
    p.style.opacity = 1;

    let currentX = startX;
    
    // Customize packet speed
    let speed = 5;
    if (type === 'ping_death') speed = 2.5; // slow dramatic giant
    if (type === 'ddos_syn') speed = 5 + Math.random() * 5; // mixed high speeds
    if (type === 'backdoor_reverse') speed = -4.5; // travels backwards!

    const fwIcon = document.getElementById('fw-icon');
    const srvIcon = document.getElementById('srv-icon');

    if (type === 'backdoor_reverse') {
      // Reverse shell starts at server node
      currentX = endX;
      p.style.left = (currentX - (p.offsetWidth/2)) + 'px';
      
      const interval = setInterval(() => {
        currentX += speed; // goes backwards (negative speed)
        p.style.left = (currentX - (packetOffsetWidth(p))) + 'px';

        // Reaches Gateway from the internal side
        if (currentX <= fwX) {
          clearInterval(interval);
          p.remove();
          
          triggerM3RippleBlock(fwX, yPos);
          stageEl.className = 'state-blocked';
          document.getElementById('client-label').innerText = 'Hôte compromis';
          document.getElementById('fw-label').innerText = 'Reverse Shell Bloqué';
          fwIcon.innerHTML = '<i data-lucide="shield-alert" style="width: 24px; height: 24px;"></i>';
          lucide.createIcons();
          
          setStep(3, 'danger', 'Intrusion bloquée (Reverse Shell)');
          setTimeout(() => {
            setStep(4, 'danger', 'Sortie Reverse Shell bloquée ✓');
          }, 300);
        }
      }, 16);
      return;
    }

    const interval = setInterval(() => {
      currentX += speed;
      p.style.left = (currentX - (packetOffsetWidth(p))) + 'px';

      // Packet reaches Firewall!
      if (currentX >= fwX) {
        clearInterval(interval);
        
        // Trigger Step 3 only for the primary sequence packet
        if (index === 0) {
          setStep(3, 'active', 'Évaluation LightGBM...');
        }

        // Action when hitting the Gateway
        setTimeout(() => {
          if (status === 'blocked') {
            p.remove();
            
            // Only trigger ripples and status block styling once (for index 0)
            if (index === 0) {
              triggerM3RippleBlock(fwX, yPos);
              stageEl.className = 'state-blocked';
              document.getElementById('client-label').innerText = 'Hôte suspect';
              document.getElementById('fw-label').innerText = 'Menace bloquée';
              fwIcon.innerHTML = '<i data-lucide="shield-alert" style="width: 24px; height: 24px;"></i>';
              
              if (type === 'teardrop') {
                // Teardrop overlapping fragment crash shake
                fwIcon.style.animation = 'srv-shake 0.15s 3';
                setTimeout(() => { fwIcon.style.animation = 'none'; }, 500);
              }
              
              lucide.createIcons();
              
              let alertText = 'Intrusion détectée';
              let decisionText = 'Flux Bloqué';
              
              if (type === 'ddos_syn') {
                alertText = `SYN Flood bloqué (${(score * 100).toFixed(0)}%)`;
                decisionText = 'DDoS Contenu ✓';
              } else if (type === 'teardrop') {
                alertText = `Fragment Teardrop (${(score * 100).toFixed(0)}%)`;
                decisionText = 'Crash évité ✓';
              } else if (type === 'ping_death') {
                alertText = `Ping de la mort (${(score * 100).toFixed(0)}%)`;
                decisionText = 'Paquet rejeté ✓';
              } else if (type === 'nmap_scan') {
                alertText = `Scan de ports (${(score * 100).toFixed(0)}%)`;
                decisionText = 'Source blacklistée ✓';
              } else if (type === 'sql_injection') {
                alertText = `SQLi bloqué (${(score * 100).toFixed(0)}%)`;
                decisionText = 'Bdd sécurisée ✓';
              } else if (type === 'buffer_overflow') {
                alertText = `Buffer Overflow (${(score * 100).toFixed(0)}%)`;
                decisionText = 'Hôte banni ✓';
              } else if (type === 'ssh_bruteforce') {
                alertText = `Brute Force SSH (${(score * 100).toFixed(0)}%)`;
                decisionText = 'IP bloquée ✓';
              } else {
                alertText = `Attaque détectée (${(score * 100).toFixed(0)}%)`;
                decisionText = `Flux Bloqué (${category.toUpperCase()})`;
              }
              
              setStep(3, 'danger', alertText);
              setTimeout(() => {
                setStep(4, 'danger', decisionText);
              }, 300);
            }
          } else {
            // Sain traffic flow
            if (index === 0) {
              stageEl.className = 'state-normal';
              document.getElementById('client-label').innerText = 'Client sécurisé';
              document.getElementById('fw-label').innerText = 'Flux inspecté';
              fwIcon.innerHTML = '<i data-lucide="shield-check" style="width: 24px; height: 24px;"></i>';
              lucide.createIcons();
              
              setStep(3, 'success', `Flux vérifié (Sain ${(score * 100).toFixed(0)}%)`);
            }
            
            animateToDest(p, fwX, endX, yPos, index, type);
          }
        }, 400);
      }
    }, 16);
  }

  function packetOffsetWidth(p) {
    return (p.offsetWidth || 14) / 2;
  }

  function animateToDest(packet, startX, endX, yPos, index, type) {
    let currentX = startX;
    const speed = 5.5;
    const srvIcon = document.getElementById('srv-icon');
    
    // Smoothly turn green as it passes through verified shield
    if (type !== 'backdoor') {
      packet.style.backgroundColor = '#146c2e';
      packet.style.boxShadow = '0 0 6px #146c2e';
      if (packet.style.width === '14px' || packet.style.width === '') {
        packet.innerHTML = '<i data-lucide="check" style="width: 8px; height: 8px; color: white; margin-top: 3px;"></i>';
        lucide.createIcons();
      } else {
        packet.innerHTML = '';
      }
    }
    
    const interval = setInterval(() => {
      currentX += speed;
      packet.style.left = (currentX - (packetOffsetWidth(packet))) + 'px';

      // Reached server
      if (currentX >= endX) {
        clearInterval(interval);
        packet.remove();
        
        // Action when arriving at server
        if (index === 0) {
          // Soft M3 server scale animation once
          srvIcon.style.transform = 'scale(1.15)';
          srvIcon.style.borderColor = '#146c2e';
            
            setStep(4, 'success', 'Flux Autorisé ✓');
            
          setTimeout(() => {
            srvIcon.style.transform = 'scale(1)';
            srvIcon.style.borderColor = '#a3cfbb';
          }, 250);
        }
      }
    }, 16);
  }

  window.addEventListener('load', () => {
    setTimeout(startSimulation, 120);
  });
</script>
</body>
</html>
"""
