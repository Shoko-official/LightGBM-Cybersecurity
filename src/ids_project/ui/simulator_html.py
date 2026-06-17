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
    left: 0;
    width: 100%;
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
  }

  @media (max-width: 600px) {
    .stage {
      flex-direction: row !important;
      justify-content: space-between !important;
      align-items: stretch !important;
      height: 310px !important;
      padding: 10px 10px !important;
      gap: 10px !important;
    }
    .nodes-container {
      width: 30% !important;
      height: 100% !important;
      flex-direction: column !important;
      justify-content: space-between !important;
      align-items: center !important;
    }
    .stepper-container {
      width: 67% !important;
      height: 100% !important;
      padding: 12px 10px !important;
    }
    .node {
      width: 100% !important;
      height: 80px !important;
      justify-content: center !important;
    }
    .node-icon-wrap {
      width: 42px !important;
      height: 42px !important;
    }
    .node-icon-wrap i {
      width: 18px !important;
      height: 18px !important;
    }
    .node-label {
      font-size: 0.65rem !important;
      bottom: 2px !important;
    }
    .step-row {
      font-size: 0.68rem !important;
      gap: 5px !important;
    }
    .step-icon-wrap {
      width: 15px !important;
      height: 15px !important;
      font-size: 0.58rem !important;
    }
    .stepper-title {
      font-size: 0.68rem !important;
      margin-bottom: 6px !important;
      padding-bottom: 4px !important;
    }
    .stepper-steps {
      gap: 5px !important;
    }
  }
</style>
</head>
<body class="state-idle">
<div class="stage" id="stage">
  <div class="nodes-container">
    <svg class="connectors-svg">
      <!-- Clean Google network track line -->
      <line x1="0" y1="0" x2="0" y2="0" stroke="#c4c7c5" stroke-width="3" id="track" />
    </svg>

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
  const lang = "__LANG__"; // 'en' | 'fr'

  const stageEl = document.body;
  const nodesContainer = document.querySelector('.nodes-container');

  function getCenters() {
    const containerRect = nodesContainer.getBoundingClientRect();
    const clientIcon = document.querySelector('.node.client .node-icon-wrap');
    const fwIcon = document.querySelector('.node.firewall .node-icon-wrap');
    const serverIcon = document.querySelector('.node.server .node-icon-wrap');

    if (!clientIcon || !fwIcon || !serverIcon || containerRect.width === 0) {
      return {
        startX: 45, fwX: 100, endX: 200, centerY: 60,
        startY: 30, fwY: 80, endY: 130, centerX: 30
      };
    }

    const clientRect = clientIcon.getBoundingClientRect();
    const fwRect = fwIcon.getBoundingClientRect();
    const serverRect = serverIcon.getBoundingClientRect();

    // Horizontal coordinates
    const startX = (clientRect.left + clientRect.right) / 2 - containerRect.left;
    const fwX = (fwRect.left + fwRect.right) / 2 - containerRect.left;
    const endX = (serverRect.left + serverRect.right) / 2 - containerRect.left;
    const centerY = (fwRect.top + fwRect.bottom) / 2 - containerRect.top;

    // Vertical coordinates
    const startY = (clientRect.top + clientRect.bottom) / 2 - containerRect.top;
    const fwY = (fwRect.top + fwRect.bottom) / 2 - containerRect.top;
    const endY = (serverRect.top + serverRect.bottom) / 2 - containerRect.top;
    const centerX = (clientRect.left + clientRect.right) / 2 - containerRect.left;

    return { startX, fwX, endX, centerY, startY, fwY, endY, centerX };
  }

  function updateTrackLine() {
    const centers = getCenters();
    const track = document.getElementById('track');
    if (!track) return;

    if (window.innerWidth <= 600) {
      track.setAttribute('x1', centers.centerX);
      track.setAttribute('x2', centers.centerX);
      track.setAttribute('y1', centers.startY);
      track.setAttribute('y2', centers.endY);
    } else {
      track.setAttribute('x1', centers.startX);
      track.setAttribute('x2', centers.endX);
      track.setAttribute('y1', centers.centerY);
      track.setAttribute('y2', centers.centerY);
    }
  }

  // Activate icons
  lucide.createIcons();

  const isEn = lang === 'en';
  const t = {
    clientLabel: isEn ? "Source" : "Source",
    clientSuspect: isEn ? "Suspicious host" : "Hôte suspect",
    clientCompromised: isEn ? "Compromised host" : "Hôte compromis",
    clientSecure: isEn ? "Secure client" : "Client sécurisé",

    fwLabel: isEn ? "IDS Gateway" : "IDS Gateway",
    fwBlocked: isEn ? "Threat blocked" : "Menace bloquée",
    fwReverseBlocked: isEn ? "Reverse Shell Blocked" : "Reverse Shell Bloqué",
    fwInspected: isEn ? "Flow inspected" : "Flux inspecté",

    serverLabel: isEn ? "Server" : "Serveur",
    analysisStatus: isEn ? "Analysis Status" : "Statut d'Analyse",

    step1Text: isEn ? "Network flow detection..." : "Détection du flux réseau...",
    step2Text: isEn ? "Metadata extraction..." : "Extraction des métadonnées...",
    step3Text: isEn ? "Model evaluation..." : "Évaluation LightGBM...",
    step4Text: isEn ? "Final security decision" : "Décision finale de sécurité",

    waitingConn: isEn ? "Waiting for connection..." : "Attente d'une connexion...",

    detectSyn: isEn ? "TCP/SYN Flow Detected" : "Flux TCP/SYN Détecté",
    extractSyn: isEn ? "Extraction: Suspected SYN Flood (480 reqs)" : "Extraction: SYN Flood suspect (480 reqs)",

    detectTeardrop: isEn ? "UDP/IP Flow Detected" : "Flux UDP/IP Détecté",
    extractTeardrop: isEn ? "Extraction: Invalid overlapping offsets" : "Extraction: Offsets superposés invalides",

    detectPing: isEn ? "ICMP/Echo Flow Detected" : "Flux ICMP/Echo Détecté",
    extractPing: isEn ? "Extraction: Giant ICMP packet (65510B)" : "Extraction: Paquet ICMP géant (65510B)",

    detectScan: isEn ? "TCP Scan Detected" : "Balayage TCP Détecté",
    extractScan: isEn ? "Extraction: Vertical stealth scan (Nmap)" : "Extraction: Scan furtif vertical (Nmap)",

    detectSqli: isEn ? "HTTP POST Request Detected" : "Requête HTTP POST Détectée",
    extractSqli: isEn ? "Extraction: Application SQL query" : "Extraction: Requête SQL applicative",

    detectBuffer: isEn ? "TCP/Telnet Connection Detected" : "Connexion TCP/Telnet Détectée",
    extractBuffer: isEn ? "Extraction: Privilege escalation shellcode" : "Extraction: Shellcode d'élévation privilèges",

    detectBrute: isEn ? "Repeated SSH Connections" : "Connexions SSH Répétées",
    extractBrute: isEn ? "Extraction: 12 authentication failures" : "Extraction: 12 échecs d'authentification",

    detectBackdoor: isEn ? "TCP/Backdoor Flow Detected" : "Flux TCP/Backdoor Détecté",
    extractBackdoor: isEn ? "Extraction: Incoming remote shell signal" : "Extraction: Signal de shell distant entrant",

    detectDefault: isEn ? "TCP/UDP flow detected" : "Flux réseau détecté",
    extractDefault: isEn ? "Extraction: Metadata extracted" : "Extraction: Métadonnées extraites",

    intrusionBlockedReverse: isEn ? "Intrusion blocked (Reverse Shell)" : "Intrusion bloquée (Reverse Shell)",
    outboundReverseBlocked: isEn ? "Outbound Reverse Shell blocked" : "Sortie Reverse Shell bloquée",

    intrusionDetected: isEn ? "Intrusion detected" : "Intrusion détectée",
    flowBlocked: isEn ? "Flow Blocked" : "Flux Bloqué",

    synBlocked: isEn ? "SYN Flood blocked" : "SYN Flood bloqué",
    ddosContained: isEn ? "DDoS Contained" : "DDoS Contenu",

    teardropBlocked: isEn ? "Teardrop Fragment" : "Fragment Teardrop",
    crashAvoided: isEn ? "Crash avoided" : "Crash évité",

    pingBlocked: isEn ? "Ping of Death" : "Ping de la mort",
    packetRejected: isEn ? "Packet rejected" : "Paquet rejeté",

    scanBlocked: isEn ? "Port scan" : "Scan de ports",
    sourceBlacklisted: isEn ? "Source blacklisted" : "Source blacklistée",

    sqliBlocked: isEn ? "SQLi blocked" : "SQLi bloqué",
    dbSecured: isEn ? "Database secured" : "Bdd sécurisée",

    bufferBlocked: isEn ? "Buffer Overflow" : "Buffer Overflow",
    hostBanned: isEn ? "Host banned" : "Hôte banni",

    bruteBlocked: isEn ? "SSH Brute Force" : "Brute Force SSH",
    ipBlocked: isEn ? "IP blocked" : "IP bloquée",

    attackDetected: isEn ? "Attack detected" : "Attaque détectée",
    flowBlockedCat: isEn ? "Flow Blocked" : "Flux Bloqué",

    flowVerified: isEn ? "Flow verified" : "Flux vérifié",
    flowAllowed: isEn ? "Flow Allowed" : "Flux Autorisé"
  };

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

  function triggerM3RippleBlock(x, y) {
    // Removed red circle propagation wave as requested by user
  }

  function resetSimulationState() {
    stageEl.className = 'state-idle';
    document.getElementById('client-label').innerText = t.clientLabel;
    document.getElementById('fw-label').innerText = t.fwLabel;

    // Set server node label and title
    document.querySelector('.node.server .node-label').innerText = t.serverLabel;
    document.querySelector('.stepper-title').innerText = t.analysisStatus;

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
      1: t.step1Text,
      2: t.step2Text,
      3: t.step3Text,
      4: t.step4Text
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
    setTimeout(updateTrackLine, 20);
  }

  function startSimulation() {
    resetSimulationState();

    if (status === 'idle') {
      setStep(1, 'active', t.waitingConn);
      return;
    }

    // Step 1: Detect
    if (scenario === 'ddos_syn') {
      setStep(1, 'success', t.detectSyn);
      setTimeout(() => setStep(2, 'success', t.extractSyn), 300);
    } else if (scenario === 'teardrop') {
      setStep(1, 'success', t.detectTeardrop);
      setTimeout(() => setStep(2, 'success', t.extractTeardrop), 300);
    } else if (scenario === 'ping_death') {
      setStep(1, 'success', t.detectPing);
      setTimeout(() => setStep(2, 'success', t.extractPing), 300);
    } else if (scenario === 'nmap_scan') {
      setStep(1, 'success', t.detectScan);
      setTimeout(() => setStep(2, 'success', t.extractScan), 300);
    } else if (scenario === 'sql_injection') {
      setStep(1, 'success', t.detectSqli);
      setTimeout(() => setStep(2, 'success', t.extractSqli), 300);
    } else if (scenario === 'buffer_overflow') {
      setStep(1, 'success', t.detectBuffer);
      setTimeout(() => setStep(2, 'success', t.extractBuffer), 300);
    } else if (scenario === 'ssh_bruteforce') {
      setStep(1, 'success', t.detectBrute);
      setTimeout(() => setStep(2, 'success', t.extractBrute), 300);
    } else if (scenario === 'backdoor') {
      setStep(1, 'success', t.detectBackdoor);
      setTimeout(() => setStep(2, 'success', t.extractBackdoor), 300);
    } else {
      setStep(1, 'success', isEn ? `Flow TCP/${protocol.toUpperCase()} detected` : `Flux TCP/${protocol.toUpperCase()} détecté`);
      setTimeout(() => setStep(2, 'success', isEn ? `Data extracted (${srcBytes} bytes)` : `Données extraites (${srcBytes} octets)`), 300);
    }

    // Step 2 activates animations at 400ms
    setTimeout(() => {
      animatePackets();
    }, 400);
  }

  function animatePackets() {
    const centers = getCenters();
    const isVert = window.innerWidth <= 600;

    const startVal = isVert ? centers.startY : centers.startX;
    const fwVal = isVert ? centers.fwY : centers.fwX;
    const endVal = isVert ? centers.endY : centers.endX;
    const orthoVal = isVert ? centers.centerX : centers.centerY;

    if (scenario === 'ddos_syn') {
      let spawned = 0;
      const interval = setInterval(() => {
        spawnSinglePacket(startVal, fwVal, endVal, orthoVal, 'ddos_syn', spawned, isVert);
        spawned++;
        if (spawned >= 90) {
          clearInterval(interval);
        }
      }, 12);
    } else if (scenario === 'teardrop') {
      for (let i = 0; i < 3; i++) {
        setTimeout(() => {
          spawnSinglePacket(startVal, fwVal, endVal, orthoVal, 'teardrop', i, isVert);
        }, i * 60);
      }
    } else if (scenario === 'nmap_scan') {
      for (let i = 0; i < 8; i++) {
        setTimeout(() => {
          spawnSinglePacket(startVal, fwVal, endVal, orthoVal, 'nmap_scan', i, isVert);
        }, i * 110);
      }
    } else if (scenario === 'ssh_bruteforce') {
      for (let i = 0; i < 8; i++) {
        setTimeout(() => {
          spawnSinglePacket(startVal, fwVal, endVal, orthoVal, 'ssh_bruteforce', i, isVert);
        }, i * 100);
      }
    } else if (scenario === 'backdoor') {
      spawnSinglePacket(startVal, fwVal, endVal, orthoVal, 'backdoor', 0, isVert);
    } else {
      spawnSinglePacket(startVal, fwVal, endVal, orthoVal, scenario, 0, isVert);
    }
  }

  function spawnSinglePacket(startVal, fwVal, endVal, orthoVal, type, index, isVert) {
    const p = document.createElement('div');
    p.className = 'packet';

    let flowPos = startVal;
    let crossPos = orthoVal;

    if (type === 'ddos_syn') {
      const size = 6 + Math.floor(Math.random() * 5);
      p.style.width = size + 'px';
      p.style.height = size + 'px';
      p.style.marginTop = -(size / 2) + 'px';
      p.style.backgroundColor = '#dc2626';
      p.style.boxShadow = '0 0 8px rgba(220,38,38,0.65)';
      const maxSpread = isVert ? 30 : (window.innerWidth <= 600 ? 40 : 72);
      crossPos = orthoVal + (Math.random() - 0.5) * maxSpread;
      p.innerHTML = '';
    } else if (type === 'teardrop') {
      p.style.width = isVert ? '12px' : '30px';
      p.style.height = isVert ? '30px' : '12px';
      p.style.borderRadius = '4px';
      p.style.marginTop = isVert ? '-15px' : '-6px';
      p.style.backgroundColor = '#d97706';
      p.style.boxShadow = '0 0 4px #d97706';

      const badge = document.createElement('div');
      badge.style.position = 'absolute';
      if (isVert) {
        badge.style.left = '18px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-20px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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
      if (isVert) {
        badge.style.left = '42px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-20px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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
      const spreadScale = isVert ? 5 : (window.innerWidth <= 600 ? 6 : 12);
      crossPos = orthoVal + (index % 3 - 1) * spreadScale;
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
      if (isVert) {
        badge.style.left = '22px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-22px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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
      if (isVert) {
        badge.style.left = '22px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-22px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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
      if (isVert) {
        badge.style.left = '18px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-20px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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
      if (isVert) {
        badge.style.left = '22px';
        badge.style.top = '50%';
        badge.style.transform = 'translateY(-50%)';
      } else {
        badge.style.top = '-22px';
        badge.style.left = '50%';
        badge.style.transform = 'translateX(-50%)';
      }
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

    if (isVert) {
      p.style.left = (crossPos - (p.offsetWidth || 14)/2) + 'px';
      p.style.top = (flowPos - (p.offsetHeight || 14)/2) + 'px';
    } else {
      p.style.left = (flowPos - (p.offsetWidth || 14)/2) + 'px';
      p.style.top = crossPos + 'px';
    }

    nodesContainer.appendChild(p);
    lucide.createIcons();
    p.style.opacity = 1;

    let currentVal = flowPos;

    let speed = 5;
    if (type === 'ping_death') speed = 2.5;
    if (type === 'ddos_syn') speed = 5 + Math.random() * 5;
    if (type === 'backdoor_reverse') speed = -4.5;

    const fwIcon = document.getElementById('fw-icon');
    const srvIcon = document.getElementById('srv-icon');

    if (type === 'backdoor_reverse') {
      currentVal = endVal;
      if (isVert) {
        p.style.top = (currentVal - (p.offsetHeight/2)) + 'px';
      } else {
        p.style.left = (currentVal - (p.offsetWidth/2)) + 'px';
      }

      const interval = setInterval(() => {
        currentVal += speed;
        if (isVert) {
          p.style.top = (currentVal - (p.offsetHeight/2)) + 'px';
        } else {
          p.style.left = (currentVal - (packetOffsetWidth(p))) + 'px';
        }

        const triggered = isVert ? (currentVal <= fwVal) : (currentVal <= fwVal);
        if (triggered) {
          clearInterval(interval);
          p.remove();

          triggerM3RippleBlock(isVert ? crossPos : fwVal, isVert ? fwVal : crossPos);
          stageEl.className = 'state-blocked';
          document.getElementById('client-label').innerText = isEn ? 'Compromised host' : 'Hôte compromis';
          document.getElementById('fw-label').innerText = isEn ? 'Reverse Shell Blocked' : 'Reverse Shell Bloqué';
          fwIcon.innerHTML = '<i data-lucide="shield-alert" style="width: 24px; height: 24px;"></i>';
          lucide.createIcons();

          setStep(3, 'danger', t.intrusionBlockedReverse);
          setTimeout(() => {
            setStep(4, 'danger', t.outboundReverseBlocked);
          }, 300);
        }
      }, 16);
      return;
    }

    const interval = setInterval(() => {
      currentVal += speed;
      if (isVert) {
        p.style.top = (currentVal - (p.offsetHeight/2)) + 'px';
      } else {
        p.style.left = (currentVal - (packetOffsetWidth(p))) + 'px';
      }

      const reachedFw = isVert ? (currentVal >= fwVal) : (currentVal >= fwVal);
      if (reachedFw) {
        clearInterval(interval);

        if (index === 0) {
          setStep(3, 'active', t.step3Text);
        }

        setTimeout(() => {
          if (status === 'blocked') {
            p.remove();

            if (index === 0) {
              triggerM3RippleBlock(isVert ? crossPos : fwVal, isVert ? fwVal : crossPos);
              stageEl.className = 'state-blocked';
              document.getElementById('client-label').innerText = t.clientSuspect;
              document.getElementById('fw-label').innerText = t.fwBlocked;
              fwIcon.innerHTML = '<i data-lucide="shield-alert" style="width: 24px; height: 24px;"></i>';

              if (type === 'teardrop') {
                fwIcon.style.animation = 'srv-shake 0.15s 3';
                setTimeout(() => { fwIcon.style.animation = 'none'; }, 500);
              }

              lucide.createIcons();

              let alertText = t.intrusionDetected;
              let decisionText = t.flowBlocked;

              if (type === 'ddos_syn') {
                alertText = isEn ? `SYN Flood blocked (${(score * 100).toFixed(0)}%)` : `SYN Flood bloqué (${(score * 100).toFixed(0)}%)`;
                decisionText = t.ddosContained;
              } else if (type === 'teardrop') {
                alertText = t.teardropBlocked;
                decisionText = t.crashAvoided;
              } else if (type === 'ping_death') {
                alertText = t.pingBlocked;
                decisionText = t.packetRejected;
              } else if (type === 'nmap_scan') {
                alertText = t.scanBlocked;
                decisionText = t.sourceBlacklisted;
              } else if (type === 'sql_injection') {
                alertText = t.sqliBlocked;
                decisionText = t.dbSecured;
              } else if (type === 'buffer_overflow') {
                alertText = t.bufferBlocked;
                decisionText = t.hostBanned;
              } else if (type === 'ssh_bruteforce') {
                alertText = t.bruteBlocked;
                decisionText = t.ipBlocked;
              } else {
                alertText = t.attackDetected;
                decisionText = t.flowBlockedCat;
              }

              setStep(3, 'danger', alertText);
              setTimeout(() => {
                setStep(4, 'danger', decisionText);
              }, 300);
            }
          } else {
            if (index === 0) {
              stageEl.className = 'state-normal';
              document.getElementById('client-label').innerText = t.clientSecure;
              document.getElementById('fw-label').innerText = t.fwInspected;
              fwIcon.innerHTML = '<i data-lucide="shield-check" style="width: 24px; height: 24px;"></i>';
              lucide.createIcons();

              setStep(3, 'success', isEn ? `Flow verified (${(score * 100).toFixed(0)}%)` : `Flux vérifié (Sain ${(score * 100).toFixed(0)}%)`);
            }

            animateToDest(p, fwVal, endVal, crossPos, index, type, isVert);
          }
        }, 400);
      }
    }, 16);
  }

  function packetOffsetWidth(p) {
    return (p.offsetWidth || 14) / 2;
  }

  function animateToDest(packet, startVal, endVal, crossPos, index, type, isVert) {
    let currentVal = startVal;
    const speed = 5.5;
    const srvIcon = document.getElementById('srv-icon');

    if (type !== 'backdoor') {
      packet.style.backgroundColor = '#146c2e';
      packet.style.boxShadow = '0 0 6px #146c2e';
      if (packet.style.width === '14px' || packet.style.width === '' || packet.style.height === '14px') {
        packet.innerHTML = '<i data-lucide="check" style="width: 8px; height: 8px; color: white; margin-top: 3px;"></i>';
        lucide.createIcons();
      } else {
        packet.innerHTML = '';
      }
    }

    const interval = setInterval(() => {
      currentVal += speed;
      if (isVert) {
        packet.style.top = (currentVal - (packet.offsetHeight/2)) + 'px';
      } else {
        packet.style.left = (currentVal - (packetOffsetWidth(packet))) + 'px';
      }

      if (currentVal >= endVal) {
        clearInterval(interval);
        packet.remove();

        if (index === 0) {
          srvIcon.style.transform = 'scale(1.15)';
          srvIcon.style.borderColor = '#146c2e';

          setStep(4, 'success', t.flowAllowed);

          setTimeout(() => {
            srvIcon.style.transform = 'scale(1)';
            srvIcon.style.borderColor = '#a3cfbb';
          }, 250);
        }
      }
    }, 16);
  }

  window.addEventListener('load', () => {
    updateTrackLine();
    setTimeout(startSimulation, 120);
  });
  window.addEventListener('resize', updateTrackLine);
</script>
</body>
</html>
"""
