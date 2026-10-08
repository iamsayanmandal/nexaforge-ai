/**
 * NexaForge — Client Application Logic
 * AI Media Processing Engine
 */

(function () {
  'use strict';

  // ── Default Backend Endpoints ─────────────────────────────────────────────
  const PRODUCTION_API_URL = 'https://nexaforge-ai-production.up.railway.app';
  const LOCAL_API_URL = 'http://localhost:8000';

  function normalizeApiUrl(url) {
    if (!url) return '';
    let clean = url.trim().replace(/\/+$/, '');
    if (!/^https?:\/\//i.test(clean)) {
      clean = 'https://' + clean;
    }
    return clean;
  }

  function resolveInitialApiUrl() {
    const saved = localStorage.getItem('nexaforge_api_url');
    if (saved) return normalizeApiUrl(saved);

    // If running on custom domain or Cloudflare Pages, always use production Railway API
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    return isLocalhost ? LOCAL_API_URL : PRODUCTION_API_URL;
  }

  // ── State Management ───────────────────────────────────────────────────────
  const state = {
    apiBaseUrl: resolveInitialApiUrl(),
    mode: 'ai',               // 'ai' | 'studio'
    studioCategory: 'img',    // 'img' | 'pdf'
    activeTool: 'img-compress',
    
    // AI Mode File
    aiFile: null,
    
    // Studio Mode Files
    studioFiles: [],
    
    // Last processed result
    lastResult: null,

    // Active session file for continuous zero-disk stream editing
    activeFileId: null,
    activeFileName: '',
  };

  // ── DOM References ────────────────────────────────────────────────────────
  const DOM = {
    // Header & Status
    apiStatusBadge: document.getElementById('api-status-badge'),
    apiStatusText: document.getElementById('api-status-text'),
    openSettingsBtn: document.getElementById('open-settings-btn'),
    toggleAssistantHeaderBtn: document.getElementById('toggle-assistant-header-btn'),

    // Mode Switcher
    tabAiMode: document.getElementById('tab-ai-mode'),
    tabStudioMode: document.getElementById('tab-studio-mode'),
    aiModePanel: document.getElementById('ai-mode-panel'),
    studioModePanel: document.getElementById('studio-mode-panel'),

    // AI Mode
    aiInstructionInput: document.getElementById('ai-instruction-input'),
    clearInstructionBtn: document.getElementById('clear-instruction-btn'),
    promptChips: document.querySelectorAll('.prompt-chip'),
    aiDropzone: document.getElementById('ai-dropzone'),
    aiFileInput: document.getElementById('ai-file-input'),
    aiDropzoneContent: document.getElementById('ai-dropzone-content'),
    aiFilePreview: document.getElementById('ai-file-preview'),
    aiThumbImg: document.getElementById('ai-thumb-img'),
    aiFileIcon: document.getElementById('ai-file-icon'),
    aiPreviewName: document.getElementById('ai-preview-name'),
    aiPreviewSize: document.getElementById('ai-preview-size'),
    aiPreviewType: document.getElementById('ai-preview-type'),
    aiRemoveFileBtn: document.getElementById('ai-remove-file-btn'),
    runAiBtn: document.getElementById('run-ai-btn'),
    aiBtnSpinner: document.getElementById('ai-btn-spinner'),
    aiBtnText: document.getElementById('ai-btn-text'),

    // Studio Mode
    subtabImg: document.getElementById('subtab-img'),
    subtabPdf: document.getElementById('subtab-pdf'),
    imageToolsGrid: document.getElementById('image-tools-grid'),
    pdfToolsGrid: document.getElementById('pdf-tools-grid'),
    toolCards: document.querySelectorAll('.tool-card'),
    studioDropzone: document.getElementById('studio-dropzone'),
    studioFileInput: document.getElementById('studio-file-input'),
    studioDropzoneContent: document.getElementById('studio-dropzone-content'),
    studioDropzoneText: document.getElementById('studio-dropzone-text'),
    studioDropzoneHint: document.getElementById('studio-dropzone-hint'),
    studioFilePreview: document.getElementById('studio-file-preview'),
    studioThumbImg: document.getElementById('studio-thumb-img'),
    studioFileIcon: document.getElementById('studio-file-icon'),
    studioPreviewName: document.getElementById('studio-preview-name'),
    studioPreviewSize: document.getElementById('studio-preview-size'),
    studioRemoveFileBtn: document.getElementById('studio-remove-file-btn'),
    runStudioBtn: document.getElementById('run-studio-btn'),
    studioBtnSpinner: document.getElementById('studio-btn-spinner'),
    studioBtnText: document.getElementById('studio-btn-text'),

    // Studio Params
    paramImgTargetKb: document.getElementById('param-img-target-kb'),
    paramImgTargetKbSlider: document.getElementById('param-img-target-kb-slider'),
    paramImgIncreaseKb: document.getElementById('param-img-increase-kb'),
    paramImgIncreaseKbSlider: document.getElementById('param-img-increase-kb-slider'),
    paramImgWidth: document.getElementById('param-img-width'),
    paramImgHeight: document.getElementById('param-img-height'),
    paramImgScale: document.getElementById('param-img-scale'),
    paramPdfTargetKb: document.getElementById('param-pdf-target-kb'),
    paramPdfTargetKbSlider: document.getElementById('param-pdf-target-kb-slider'),
    paramPdfIncreaseKb: document.getElementById('param-pdf-increase-kb'),
    paramPdfIncreaseKbSlider: document.getElementById('param-pdf-increase-kb-slider'),
    paramPdfStartPage: document.getElementById('param-pdf-start-page'),
    paramPdfEndPage: document.getElementById('param-pdf-end-page'),
    paramPdfPagesList: document.getElementById('param-pdf-pages-list'),

    // Progress & Results
    processingOverlay: document.getElementById('processing-overlay'),
    processingTitle: document.getElementById('processing-status-title'),
    processingSubtitle: document.getElementById('processing-status-step'),
    resultCard: document.getElementById('result-card'),
    resultFilename: document.getElementById('result-filename-display'),
    aiResponseBox: document.getElementById('ai-response-box'),
    resultReplyText: document.getElementById('result-reply-text'),
    resultVerificationBadge: document.getElementById('result-verification-badge'),
    verificationTargetVal: document.getElementById('verification-target-val'),
    verificationAuditVal: document.getElementById('verification-audit-val'),
    metricOrigSize: document.getElementById('metric-orig-size'),
    metricNewSize: document.getElementById('metric-new-size'),
    metricReduction: document.getElementById('metric-reduction'),
    downloadResultLink: document.getElementById('download-result-link'),
    processAnotherBtn: document.getElementById('process-another-btn'),

    // Chained Stream Editing
    chainedStreamBox: document.getElementById('chained-stream-box'),
    streamStatusBadge: document.getElementById('stream-status-badge'),
    chainedChipsList: document.getElementById('chained-chips-list'),
    chainedForm: document.getElementById('chained-form'),
    chainedInput: document.getElementById('chained-input'),
    chainedSubmitBtn: document.getElementById('chained-submit-btn'),
    chainedSpinner: document.getElementById('chained-spinner'),
    chainedBtnText: document.getElementById('chained-btn-text'),

    // Diagnostics Modal
    diagnosticsModal: document.getElementById('diagnostics-modal'),
    closeDiagnosticsBtn: document.getElementById('close-diagnostics-btn'),
    closeDiagnosticsBtn2: document.getElementById('close-diagnostics-btn-2'),
    refreshDiagnosticsBtn: document.getElementById('refresh-diagnostics-btn'),
    servicesList: document.getElementById('services-list'),

    // Assistant Drawer
    assistantDrawer: document.getElementById('assistant-drawer'),
    closeAssistantBtn: document.getElementById('close-assistant-btn'),
    floatingAssistantBtn: document.getElementById('floating-assistant-btn'),
    chatMessages: document.getElementById('chat-messages'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),

    // Settings Modal
    settingsModal: document.getElementById('settings-modal'),
    closeSettingsBtn: document.getElementById('close-settings-btn'),
    apiBaseUrlInput: document.getElementById('api-base-url-input'),
    testApiBtn: document.getElementById('test-api-btn'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    apiTestResult: document.getElementById('api-test-result'),

    // Toasts
    toastContainer: document.getElementById('toast-container'),
  };

  // ── Helpers ───────────────────────────────────────────────────────────────
  function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    DOM.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = '0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ── System Diagnostics & Recruiter Telemetry ──────────────────────────────
  async function checkSystemDiagnostics(openModal = false) {
    if (DOM.refreshDiagnosticsBtn && openModal) {
      DOM.refreshDiagnosticsBtn.disabled = true;
    }

    try {
      let res;
      try {
        res = await fetch(`${state.apiBaseUrl}/api/system/status`, { method: 'GET' });
        if (!res.ok && state.apiBaseUrl === LOCAL_API_URL) {
          throw new Error('Local server non-200');
        }
      } catch (localErr) {
        // Auto-fallback: if local backend is offline, route to live Railway cloud backend
        if (state.apiBaseUrl === LOCAL_API_URL) {
          state.apiBaseUrl = PRODUCTION_API_URL;
          res = await fetch(`${state.apiBaseUrl}/api/system/status`, { method: 'GET' });
        } else {
          throw localErr;
        }
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const comps = data.components || {};

      const isFastApiOnline = comps.api_server?.status === 'online';
      const isLlmOnline = comps.llm_engine?.status === 'online';
      const isChromaOnline = comps.vector_rag?.status === 'online';

      // Badge status logic
      if (isFastApiOnline && isLlmOnline && isChromaOnline) {
        DOM.apiStatusBadge.className = 'status-badge-btn online';
        DOM.apiStatusText.textContent = 'All Systems Live';
      } else if (isFastApiOnline) {
        DOM.apiStatusBadge.className = 'status-badge-btn warning';
        DOM.apiStatusText.textContent = 'Degraded Services';
      } else {
        DOM.apiStatusBadge.className = 'status-badge-btn offline';
        DOM.apiStatusText.textContent = 'Backend Offline';
      }

      // Populate Diagnostics Modal if requested or open
      if (DOM.servicesList && (openModal || DOM.diagnosticsModal.style.display === 'grid')) {
        renderDiagnosticsModal(comps, data);
      }

      if (openModal && DOM.diagnosticsModal) {
        DOM.diagnosticsModal.style.display = 'grid';
      }
    } catch (err) {
      DOM.apiStatusBadge.className = 'status-badge-btn offline';
      DOM.apiStatusText.textContent = 'Backend Offline';

      if (DOM.servicesList && openModal) {
        DOM.servicesList.innerHTML = `
          <div class="service-item-card">
            <div class="service-info">
              <span class="service-name">FastAPI Core Engine</span>
              <span class="service-meta">Could not establish connection to ${escapeHtml(state.apiBaseUrl)}</span>
            </div>
            <span class="service-badge-pill offline">
              <span class="status-dot red"></span>
              <span>Offline</span>
            </span>
          </div>
        `;
        if (DOM.diagnosticsModal) DOM.diagnosticsModal.style.display = 'grid';
      }
    } finally {
      if (DOM.refreshDiagnosticsBtn) {
        DOM.refreshDiagnosticsBtn.disabled = false;
      }
    }
  }

  function renderDiagnosticsModal(comps, data) {
    if (!DOM.servicesList) return;
    const list = [
      {
        name: 'FastAPI Backend Engine',
        meta: 'ASGI / Uvicorn • Zero-Disk Memory Pipeline (RAM Safe)',
        status: comps.api_server?.status === 'online' ? 'online' : 'offline',
        label: comps.api_server?.status === 'online' ? 'Online' : 'Offline',
      },
      {
        name: 'Groq GPT-OSS Inference Engine',
        meta: `Model: ${comps.llm_engine?.model || 'openai/gpt-oss-20b'} • Provider: ${comps.llm_engine?.provider || 'Groq Cloud'}`,
        status: comps.llm_engine?.status === 'online' ? 'online' : 'warning',
        label: comps.llm_engine?.status === 'online' ? 'Connected' : 'Missing Key',
      },
      {
        name: 'ChromaDB Vector Store (RAG)',
        meta: `${comps.vector_rag?.chunks_indexed || 0} Knowledge Chunks • ${comps.vector_rag?.embedding_model || 'sentence-transformers'}`,
        status: comps.vector_rag?.status === 'online' ? 'online' : 'warning',
        label: comps.vector_rag?.status === 'online' ? 'Connected' : 'Degraded',
      },
      {
        name: 'LangSmith Observability & Tracing',
        meta: `Project: ${comps.observability?.project || 'nexaforge'} • Real-Time Pipeline Tracing`,
        status: comps.observability?.status === 'active' ? 'online' : 'warning',
        label: comps.observability?.status === 'active' ? 'Active' : 'Unconfigured',
      },
    ];

    DOM.servicesList.innerHTML = list.map(item => `
      <div class="service-item-card">
        <div class="service-info">
          <span class="service-name">${item.name}</span>
          <span class="service-meta">${item.meta}</span>
        </div>
        <span class="service-badge-pill ${item.status}">
          <span class="status-dot ${item.status === 'online' ? 'green' : item.status === 'warning' ? 'amber' : 'red'}"></span>
          <span>${item.label}</span>
        </span>
      </div>
    `).join('');
  }

  // Diagnostics Modal Triggers
  if (DOM.apiStatusBadge) {
    DOM.apiStatusBadge.addEventListener('click', () => checkSystemDiagnostics(true));
  }
  if (DOM.closeDiagnosticsBtn) {
    DOM.closeDiagnosticsBtn.addEventListener('click', () => { DOM.diagnosticsModal.style.display = 'none'; });
  }
  if (DOM.closeDiagnosticsBtn2) {
    DOM.closeDiagnosticsBtn2.addEventListener('click', () => { DOM.diagnosticsModal.style.display = 'none'; });
  }
  if (DOM.refreshDiagnosticsBtn) {
    DOM.refreshDiagnosticsBtn.addEventListener('click', () => checkSystemDiagnostics(true));
  }

  // ── Mode Switcher ─────────────────────────────────────────────────────────
  DOM.tabAiMode.addEventListener('click', () => switchMode('ai'));
  DOM.tabStudioMode.addEventListener('click', () => switchMode('studio'));

  function switchMode(newMode) {
    state.mode = newMode;
    if (newMode === 'ai') {
      DOM.tabAiMode.classList.add('active');
      DOM.tabStudioMode.classList.remove('active');
      DOM.aiModePanel.style.display = 'block';
      DOM.studioModePanel.style.display = 'none';
    } else {
      DOM.tabStudioMode.classList.add('active');
      DOM.tabAiMode.classList.remove('active');
      DOM.studioModePanel.style.display = 'block';
      DOM.aiModePanel.style.display = 'none';
      updateStudioUI();
    }
    DOM.resultCard.style.display = 'none';
  }

  // ── AI Auto-Pilot Handlers ────────────────────────────────────────────────
  DOM.aiInstructionInput.addEventListener('input', () => {
    DOM.clearInstructionBtn.style.display = DOM.aiInstructionInput.value ? 'block' : 'none';
    updateAiExecuteButtonState();
  });

  DOM.clearInstructionBtn.addEventListener('click', () => {
    DOM.aiInstructionInput.value = '';
    DOM.clearInstructionBtn.style.display = 'none';
    updateAiExecuteButtonState();
    DOM.aiInstructionInput.focus();
  });

  DOM.promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      DOM.aiInstructionInput.value = chip.dataset.prompt;
      DOM.clearInstructionBtn.style.display = 'block';
      updateAiExecuteButtonState();
      DOM.aiInstructionInput.focus();
    });
  });

  // AI Dropzone
  DOM.aiDropzone.addEventListener('click', (e) => {
    if (e.target.closest('#ai-remove-file-btn')) return;
    DOM.aiFileInput.click();
  });

  DOM.aiFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleAiFileSelection(file);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    DOM.aiDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.aiDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    DOM.aiDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.aiDropzone.classList.remove('dragover');
    });
  });

  DOM.aiDropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleAiFileSelection(file);
  });

  function handleAiFileSelection(file) {
    if (file.size > 10 * 1024 * 1024) {
      showToast('File exceeds 10MB limit. Please upload a smaller file.', 'error');
      return;
    }
    state.aiFile = file;
    DOM.aiDropzoneContent.style.display = 'none';
    DOM.aiFilePreview.style.display = 'flex';
    DOM.aiPreviewName.textContent = file.name;
    DOM.aiPreviewSize.textContent = formatBytes(file.size);

    if (file.type.startsWith('image/')) {
      DOM.aiPreviewType.textContent = 'Image';
      DOM.aiFileIcon.style.display = 'none';
      DOM.aiThumbImg.style.display = 'block';
      const reader = new FileReader();
      reader.onload = (ev) => {
        DOM.aiThumbImg.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    } else {
      DOM.aiPreviewType.textContent = 'PDF Document';
      DOM.aiThumbImg.style.display = 'none';
      DOM.aiFileIcon.style.display = 'block';
    }

    updateAiExecuteButtonState();
    DOM.resultCard.style.display = 'none';
  }

  DOM.aiRemoveFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.aiFile = null;
    DOM.aiFileInput.value = '';
    DOM.aiDropzoneContent.style.display = 'block';
    DOM.aiFilePreview.style.display = 'none';
    DOM.aiThumbImg.src = '';
    updateAiExecuteButtonState();
  });

  function updateAiExecuteButtonState() {
    const hasInstruction = !!DOM.aiInstructionInput.value.trim();
    const hasFile = !!state.aiFile;
    DOM.runAiBtn.disabled = !(hasInstruction && hasFile);
  }

  // AI Pipeline Execution with Self-Verification
  DOM.runAiBtn.addEventListener('click', async () => {
    if (!state.aiFile || !DOM.aiInstructionInput.value.trim()) return;

    const instruction = DOM.aiInstructionInput.value.trim();
    const formData = new FormData();
    formData.append('file', state.aiFile);
    formData.append('instruction', instruction);

    showProcessing('Orchestrating LangGraph Agent...', 'Analyzing requirements and executing deterministic pipeline');
    DOM.runAiBtn.disabled = true;
    DOM.aiBtnSpinner.style.display = 'inline-block';
    DOM.aiBtnText.textContent = 'Agent Executing...';

    try {
      const response = await fetch(`${state.apiBaseUrl}/api/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errMessage = 'Processing failed';
        try {
          const errData = await response.json();
          errMessage = errData.detail || errMessage;
        } catch (_) {}
        throw new Error(errMessage);
      }

      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const resData = await response.json();
        const fileRes = await fetch(`${state.apiBaseUrl}${resData.download_url}`);
        const blob = await fileRes.blob();
        handleSuccessfulResult(blob, resData.filename, resData.metrics.original_size, resData);
        showToast('Operation verified and completed successfully.', 'success');
      } else {
        let outFilename = 'processed_file';
        const disposition = response.headers.get('content-disposition');
        if (disposition && disposition.includes('filename=')) {
          const match = disposition.match(/filename="?([^"]+)"?/);
          if (match && match[1]) outFilename = match[1];
        }
        const blob = await response.blob();
        handleSuccessfulResult(blob, outFilename, state.aiFile.size);
        showToast('Operation completed successfully.', 'success');
      }
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      hideProcessing();
      DOM.runAiBtn.disabled = false;
      DOM.aiBtnSpinner.style.display = 'none';
      DOM.aiBtnText.textContent = 'Execute AI Pipeline';
    }
  });

  // ── Studio Tools Handlers ─────────────────────────────────────────────────
  DOM.subtabImg.addEventListener('click', () => {
    state.studioCategory = 'img';
    DOM.subtabImg.classList.add('active');
    DOM.subtabPdf.classList.remove('active');
    DOM.imageToolsGrid.style.display = 'grid';
    DOM.pdfToolsGrid.style.display = 'none';
    selectTool('img-compress');
  });

  DOM.subtabPdf.addEventListener('click', () => {
    state.studioCategory = 'pdf';
    DOM.subtabPdf.classList.add('active');
    DOM.subtabImg.classList.remove('active');
    DOM.imageToolsGrid.style.display = 'none';
    DOM.pdfToolsGrid.style.display = 'grid';
    selectTool('pdf-compress');
  });

  DOM.toolCards.forEach(card => {
    card.addEventListener('click', () => {
      selectTool(card.dataset.tool);
    });
  });

  function selectTool(toolName) {
    state.activeTool = toolName;
    DOM.toolCards.forEach(c => {
      c.classList.toggle('active', c.dataset.tool === toolName);
    });

    // Hide all param panels
    const paramPanels = [
      'params-img-compress', 'params-img-increase', 'params-img-resize', 'params-img-convert', 'params-img-rotate',
      'params-pdf-compress', 'params-pdf-increase', 'params-pdf-split', 'params-pdf-extract'
    ];
    paramPanels.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });

    // Show active param panel
    const targetPanel = document.getElementById(`params-${toolName}`);
    if (targetPanel) targetPanel.style.display = 'block';

    updateStudioUI();
  }

  function updateStudioUI() {
    const isMultiFile = state.activeTool === 'pdf-merge' || state.activeTool === 'pdf-images-to-pdf';
    DOM.studioFileInput.multiple = isMultiFile;
    if (isMultiFile) {
      DOM.studioDropzoneText.textContent = 'Select multiple files to merge/combine';
      DOM.studioDropzoneHint.textContent = state.activeTool === 'pdf-merge' ? 'Select 2 or more PDF files' : 'Select images (JPG, PNG, WEBP)';
    } else {
      DOM.studioDropzoneText.textContent = 'Select a file for this operation';
      DOM.studioDropzoneHint.textContent = state.activeTool.startsWith('img') ? 'Upload JPG, PNG, or WEBP' : 'Upload PDF document';
    }
    updateStudioExecuteButton();
  }

  // Sliders sync
  if (DOM.paramImgTargetKbSlider && DOM.paramImgTargetKb) {
    DOM.paramImgTargetKbSlider.addEventListener('input', (e) => DOM.paramImgTargetKb.value = e.target.value);
    DOM.paramImgTargetKb.addEventListener('input', (e) => DOM.paramImgTargetKbSlider.value = e.target.value);
  }
  if (DOM.paramImgIncreaseKbSlider && DOM.paramImgIncreaseKb) {
    DOM.paramImgIncreaseKbSlider.addEventListener('input', (e) => DOM.paramImgIncreaseKb.value = e.target.value);
    DOM.paramImgIncreaseKb.addEventListener('input', (e) => DOM.paramImgIncreaseKbSlider.value = e.target.value);
  }
  if (DOM.paramPdfTargetKbSlider && DOM.paramPdfTargetKb) {
    DOM.paramPdfTargetKbSlider.addEventListener('input', (e) => DOM.paramPdfTargetKb.value = e.target.value);
    DOM.paramPdfTargetKb.addEventListener('input', (e) => DOM.paramPdfTargetKbSlider.value = e.target.value);
  }
  if (DOM.paramPdfIncreaseKbSlider && DOM.paramPdfIncreaseKb) {
    DOM.paramPdfIncreaseKbSlider.addEventListener('input', (e) => DOM.paramPdfIncreaseKb.value = e.target.value);
    DOM.paramPdfIncreaseKb.addEventListener('input', (e) => DOM.paramPdfIncreaseKbSlider.value = e.target.value);
  }

  // Studio Dropzone
  DOM.studioDropzone.addEventListener('click', (e) => {
    if (e.target.closest('#studio-remove-file-btn')) return;
    DOM.studioFileInput.click();
  });

  DOM.studioFileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length) handleStudioFiles(files);
  });

  function handleStudioFiles(files) {
    state.studioFiles = files;
    DOM.studioDropzoneContent.style.display = 'none';
    DOM.studioFilePreview.style.display = 'flex';

    if (files.length === 1) {
      const file = files[0];
      DOM.studioPreviewName.textContent = file.name;
      DOM.studioPreviewSize.textContent = formatBytes(file.size);
      if (file.type.startsWith('image/')) {
        DOM.studioFileIcon.style.display = 'none';
        DOM.studioThumbImg.style.display = 'block';
        const reader = new FileReader();
        reader.onload = (ev) => DOM.studioThumbImg.src = ev.target.result;
        reader.readAsDataURL(file);
      } else {
        DOM.studioThumbImg.style.display = 'none';
        DOM.studioFileIcon.style.display = 'block';
      }
    } else {
      DOM.studioPreviewName.textContent = `${files.length} files selected`;
      const totalSize = files.reduce((acc, f) => acc + f.size, 0);
      DOM.studioPreviewSize.textContent = `Total: ${formatBytes(totalSize)}`;
      DOM.studioThumbImg.style.display = 'none';
      DOM.studioFileIcon.style.display = 'block';
    }

    updateStudioExecuteButton();
    DOM.resultCard.style.display = 'none';
  }

  DOM.studioRemoveFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.studioFiles = [];
    DOM.studioFileInput.value = '';
    DOM.studioDropzoneContent.style.display = 'block';
    DOM.studioFilePreview.style.display = 'none';
    updateStudioExecuteButton();
  });

  function updateStudioExecuteButton() {
    const isMulti = state.activeTool === 'pdf-merge' || state.activeTool === 'pdf-images-to-pdf';
    if (isMulti) {
      DOM.runStudioBtn.disabled = state.studioFiles.length < (state.activeTool === 'pdf-merge' ? 2 : 1);
    } else {
      DOM.runStudioBtn.disabled = state.studioFiles.length !== 1;
    }
  }

  // Studio Execution
  DOM.runStudioBtn.addEventListener('click', async () => {
    if (!state.studioFiles.length) return;

    const tool = state.activeTool;
    const formData = new FormData();
    let endpoint = '';
    let origSize = 0;

    if (tool === 'img-compress') {
      endpoint = `${state.apiBaseUrl}/api/image/compress`;
      formData.append('file', state.studioFiles[0]);
      formData.append('target_kb', DOM.paramImgTargetKb.value || 100);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'img-increase') {
      endpoint = `${state.apiBaseUrl}/api/image/increase`;
      formData.append('file', state.studioFiles[0]);
      formData.append('target_kb', DOM.paramImgIncreaseKb?.value || 100);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'img-resize') {
      endpoint = `${state.apiBaseUrl}/api/image/resize`;
      formData.append('file', state.studioFiles[0]);
      if (DOM.paramImgWidth.value) formData.append('width', DOM.paramImgWidth.value);
      if (DOM.paramImgHeight.value) formData.append('height', DOM.paramImgHeight.value);
      if (DOM.paramImgScale.value) formData.append('scale_percent', DOM.paramImgScale.value);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'img-convert') {
      endpoint = `${state.apiBaseUrl}/api/image/convert`;
      formData.append('file', state.studioFiles[0]);
      const fmt = document.querySelector('input[name="convert-fmt"]:checked')?.value || 'WEBP';
      formData.append('target_format', fmt);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'img-rotate') {
      endpoint = `${state.apiBaseUrl}/api/image/rotate`;
      formData.append('file', state.studioFiles[0]);
      const deg = document.querySelector('input[name="rotate-deg"]:checked')?.value || '90';
      formData.append('degrees', deg);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'pdf-compress') {
      endpoint = `${state.apiBaseUrl}/api/pdf/compress`;
      formData.append('file', state.studioFiles[0]);
      formData.append('target_kb', DOM.paramPdfTargetKb.value || 300);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'pdf-increase') {
      endpoint = `${state.apiBaseUrl}/api/pdf/increase`;
      formData.append('file', state.studioFiles[0]);
      formData.append('target_kb', DOM.paramPdfIncreaseKb?.value || 200);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'pdf-merge') {
      endpoint = `${state.apiBaseUrl}/api/pdf/merge`;
      state.studioFiles.forEach(f => formData.append('files', f));
      origSize = state.studioFiles.reduce((a, b) => a + b.size, 0);
    } else if (tool === 'pdf-split') {
      endpoint = `${state.apiBaseUrl}/api/pdf/split`;
      formData.append('file', state.studioFiles[0]);
      formData.append('start_page', DOM.paramPdfStartPage.value || 1);
      if (DOM.paramPdfEndPage.value) formData.append('end_page', DOM.paramPdfEndPage.value);
      origSize = state.studioFiles[0].size;
    } else if (tool === 'pdf-extract') {
      endpoint = `${state.apiBaseUrl}/api/pdf/extract-pages`;
      formData.append('file', state.studioFiles[0]);
      formData.append('pages', DOM.paramPdfPagesList.value || '1');
      origSize = state.studioFiles[0].size;
    } else if (tool === 'pdf-images-to-pdf') {
      endpoint = `${state.apiBaseUrl}/api/pdf/images-to-pdf`;
      state.studioFiles.forEach(f => formData.append('files', f));
      origSize = state.studioFiles.reduce((a, b) => a + b.size, 0);
    }

    showProcessing(`Executing ${tool.replace('-', ' ').toUpperCase()}`, 'Deterministic processing in memory');
    DOM.runStudioBtn.disabled = true;
    DOM.studioBtnSpinner.style.display = 'inline-block';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errMessage = 'Studio tool error';
        try {
          const errData = await response.json();
          errMessage = errData.detail || errMessage;
        } catch (_) {}
        throw new Error(errMessage);
      }

      let outFilename = 'processed_file';
      const disposition = response.headers.get('content-disposition');
      if (disposition && disposition.includes('filename=')) {
        const match = disposition.match(/filename="?([^"]+)"?/);
        if (match && match[1]) outFilename = match[1];
      }

      const blob = await response.blob();
      handleSuccessfulResult(blob, outFilename, origSize);
      showToast('Tool execution completed!', 'success');
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      hideProcessing();
      DOM.runStudioBtn.disabled = false;
      DOM.studioBtnSpinner.style.display = 'none';
    }
  });

  // ── Results & Progress ────────────────────────────────────────────────────
  function showProcessing(title, step) {
    DOM.processingTitle.textContent = title;
    DOM.processingSubtitle.textContent = step;
    DOM.processingOverlay.style.display = 'flex';
  }

  function hideProcessing() {
    DOM.processingOverlay.style.display = 'none';
  }

  function handleSuccessfulResult(blob, filename, origSizeBytes, aiReport = null) {
    if (state.lastResult && state.lastResult.url) {
      URL.revokeObjectURL(state.lastResult.url);
    }

    const downloadUrl = URL.createObjectURL(blob);
    state.lastResult = { blob, filename, url: downloadUrl };

    DOM.resultFilename.textContent = filename;
    DOM.downloadResultLink.href = downloadUrl;
    DOM.downloadResultLink.download = filename;

    DOM.metricOrigSize.textContent = formatBytes(origSizeBytes);
    DOM.metricNewSize.textContent = formatBytes(blob.size);

    if (aiReport && DOM.aiResponseBox) {
      DOM.aiResponseBox.style.display = 'block';
      DOM.resultReplyText.textContent = aiReport.reply || 'Processing complete.';
      const isPassed = aiReport.verification?.status === 'PASSED';
      DOM.resultVerificationBadge.textContent = isPassed ? 'Verification Passed' : 'Verified with Warnings';
      DOM.resultVerificationBadge.className = isPassed ? 'verification-badge' : 'verification-badge failed';
      DOM.verificationTargetVal.textContent = aiReport.verification?.target_constraint ? `Constraint: ${aiReport.verification.target_constraint}` : '';
      DOM.verificationAuditVal.textContent = aiReport.verification?.details ? `Audit: ${aiReport.verification.details}` : '';
    } else if (DOM.aiResponseBox) {
      DOM.aiResponseBox.style.display = 'none';
    }

    // Set active in-memory session file for continuous zero-disk stream editing
    if (aiReport && aiReport.session_file_id) {
      state.activeFileId = aiReport.session_file_id;
      state.activeFileName = filename;
      if (DOM.chainedStreamBox) {
        DOM.chainedStreamBox.style.display = 'block';
        if (DOM.streamStatusBadge) {
          DOM.streamStatusBadge.textContent = `RAM Active: ${filename}`;
        }
      }
    } else if (DOM.chainedStreamBox && !state.activeFileId) {
      DOM.chainedStreamBox.style.display = 'none';
    }

    if (origSizeBytes > 0) {
      const diff = origSizeBytes - blob.size;
      const pct = Math.round((diff / origSizeBytes) * 100);
      if (pct > 0) {
        DOM.metricReduction.textContent = `-${pct}% Saved`;
        DOM.metricReduction.style.color = 'var(--accent-success)';
      } else if (pct < 0) {
        DOM.metricReduction.textContent = `+${Math.abs(pct)}% Expanded`;
        DOM.metricReduction.style.color = 'var(--accent-cyan)';
      } else {
        DOM.metricReduction.textContent = 'Exact Size';
        DOM.metricReduction.style.color = 'var(--text-muted)';
      }
    } else {
      DOM.metricReduction.textContent = 'Generated';
    }

    DOM.resultCard.style.display = 'block';
    DOM.resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  DOM.processAnotherBtn.addEventListener('click', () => {
    DOM.resultCard.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // ── Continuous In-Memory Stream Editing ───────────────────────────────────
  if (DOM.chainedChipsList) {
    DOM.chainedChipsList.addEventListener('click', (e) => {
      const chip = e.target.closest('.chained-chip');
      if (chip && chip.dataset.chainPrompt) {
        const prompt = chip.dataset.chainPrompt;
        if (DOM.chainedInput) DOM.chainedInput.value = prompt;
        executeChainedStreamEdit(prompt);
      }
    });
  }

  if (DOM.chainedForm) {
    DOM.chainedForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const instruction = DOM.chainedInput ? DOM.chainedInput.value.trim() : '';
      if (!instruction) return;
      executeChainedStreamEdit(instruction);
    });
  }

  async function executeChainedStreamEdit(instruction) {
    if (!state.activeFileId) {
      showToast('No active file in session. Please process a file first.', 'error');
      return;
    }

    if (DOM.chainedSubmitBtn) DOM.chainedSubmitBtn.disabled = true;
    if (DOM.chainedSpinner) DOM.chainedSpinner.style.display = 'inline-block';
    if (DOM.chainedBtnText) DOM.chainedBtnText.textContent = 'Transforming...';

    showProcessing(
      'Streaming In-Memory Transformation...',
      `Executing "${instruction}" on ${state.activeFileName || 'active file'} without re-upload`
    );

    const formData = new FormData();
    formData.append('file_id', state.activeFileId);
    formData.append('instruction', instruction);

    try {
      const response = await fetch(`${state.apiBaseUrl}/api/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errMessage = 'Stream operation failed';
        try {
          const errData = await response.json();
          errMessage = errData.detail || errMessage;
        } catch (_) {}
        throw new Error(errMessage);
      }

      const resData = await response.json();
      const fileRes = await fetch(`${state.apiBaseUrl}${resData.download_url}`);
      const newBlob = await fileRes.blob();

      if (DOM.chainedInput) DOM.chainedInput.value = '';

      handleSuccessfulResult(
        newBlob, 
        resData.filename, 
        resData.metrics?.original_size || state.lastResult?.blob?.size || 0, 
        resData
      );
      showToast(`Transformed: ${resData.filename}`, 'success');
    } catch (err) {
      showToast(`Stream Error: ${err.message}`, 'error');
    } finally {
      hideProcessing();
      if (DOM.chainedSubmitBtn) DOM.chainedSubmitBtn.disabled = false;
      if (DOM.chainedSpinner) DOM.chainedSpinner.style.display = 'none';
      if (DOM.chainedBtnText) DOM.chainedBtnText.textContent = 'Apply Edit';
    }
  }

  // ── RAG Assistant Chatbot ─────────────────────────────────────────────────
  function toggleAssistantDrawer() {
    DOM.assistantDrawer.classList.toggle('open');
    if (DOM.assistantDrawer.classList.contains('open')) {
      DOM.chatInput.focus();
    }
  }

  DOM.floatingAssistantBtn.addEventListener('click', toggleAssistantDrawer);
  DOM.toggleAssistantHeaderBtn.addEventListener('click', toggleAssistantDrawer);
  DOM.closeAssistantBtn.addEventListener('click', () => DOM.assistantDrawer.classList.remove('open'));

  DOM.chatMessages.addEventListener('click', (e) => {
    const chip = e.target.closest('.rag-chip');
    if (chip) {
      const q = chip.dataset.q;
      sendChatMessage(q);
    }
  });

  DOM.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = DOM.chatInput.value.trim();
    if (!text) return;
    DOM.chatInput.value = '';
    sendChatMessage(text);
  });

  async function sendChatMessage(question) {
    // Append user message
    appendChatBubble(question, 'user');

    // Append loading bot message
    const botBubble = appendChatBubble('Thinking...', 'bot');

    try {
      const res = await fetch(`${state.apiBaseUrl}/api/assistant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) throw new Error('Assistant API error');
      const data = await res.json();

      let formattedHtml = renderMarkdownCleanly(data.answer);
      if (data.sources && data.sources.length) {
        const cleanSources = data.sources.map(s => `<code>${escapeHtml(s)}</code>`).join(' ');
        formattedHtml += `<div class="chat-sources">Knowledge Sources: ${cleanSources}</div>`;
      }
      botBubble.innerHTML = formattedHtml;
    } catch (err) {
      botBubble.innerHTML = `<p>Could not connect to assistant (${err.message}). Please ensure the backend is active.</p>`;
    }

    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
  }

  function appendChatBubble(text, role) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    if (role === 'bot') {
      bubble.innerHTML = renderMarkdownCleanly(text);
    } else {
      bubble.innerHTML = `<p>${escapeHtml(text)}</p>`;
    }
    DOM.chatMessages.appendChild(bubble);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    return bubble;
  }

  function renderMarkdownCleanly(raw) {
    if (!raw) return '';
    let text = raw.trim();

    // 1. Strip raw markdown horizontal dividers (***, ---)
    text = text.replace(/^[\*\-_]{3,}\s*$/gm, '');

    // 2. Parse Markdown Tables if present (| Col 1 | Col 2 |)
    const lines = text.split('\n');
    let inTable = false;
    let tableHtml = '';
    const processedLines = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('|') && line.endsWith('|')) {
        const cells = line.split('|').slice(1, -1).map(c => c.trim());
        // Skip separator line (|---|---|)
        if (cells.every(c => /^:?-+:?$/.test(c))) {
          continue;
        }
        if (!inTable) {
          inTable = true;
          tableHtml = '<table class="chat-table"><thead><tr>';
          cells.forEach(c => { tableHtml += `<th>${inlineFormat(c)}</th>`; });
          tableHtml += '</tr></thead><tbody>';
        } else {
          tableHtml += '<tr>';
          cells.forEach(c => { tableHtml += `<td>${inlineFormat(c)}</td>`; });
          tableHtml += '</tr>';
        }
      } else {
        if (inTable) {
          tableHtml += '</tbody></table>';
          processedLines.push(tableHtml);
          inTable = false;
          tableHtml = '';
        }
        processedLines.push(lines[i]);
      }
    }
    if (inTable) {
      tableHtml += '</tbody></table>';
      processedLines.push(tableHtml);
    }

    // 3. Process Headings, Lists, Paragraphs
    let output = '';
    let inList = false;

    for (let i = 0; i < processedLines.length; i++) {
      const l = processedLines[i];
      if (l.startsWith('<table')) {
        if (inList) { output += '</ul>'; inList = false; }
        output += l;
        continue;
      }

      const trimmed = l.trim();
      if (!trimmed) {
        if (inList) { output += '</ul>'; inList = false; }
        continue;
      }

      // Headings (### Title -> <h4>Title</h4> without any #)
      const headMatch = trimmed.match(/^#{1,6}\s+(.+)$/);
      if (headMatch) {
        if (inList) { output += '</ul>'; inList = false; }
        output += `<h4>${inlineFormat(headMatch[1])}</h4>`;
        continue;
      }

      // Bullet lists (- or * or •)
      const listMatch = trimmed.match(/^[-*•]\s+(.+)$/);
      if (listMatch) {
        if (!inList) {
          inList = true;
          output += '<ul class="chat-list">';
        }
        output += `<li>${inlineFormat(listMatch[1])}</li>`;
        continue;
      }

      // Ordered lists (1. 2.)
      const numMatch = trimmed.match(/^\d+\.\s+(.+)$/);
      if (numMatch) {
        if (!inList) {
          inList = true;
          output += '<ol class="chat-list">';
        }
        output += `<li>${inlineFormat(numMatch[1])}</li>`;
        continue;
      }

      // Normal paragraph
      if (inList) { output += '</ul>'; inList = false; }
      output += `<p>${inlineFormat(trimmed)}</p>`;
    }

    if (inList) { output += '</ul>'; }
    return output;
  }

  function inlineFormat(str) {
    if (!str) return '';
    let s = escapeHtml(str);
    // Code ticks
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold **text**
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic *text*
    s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return s;
  }

  function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  }

  // ── Settings Modal ────────────────────────────────────────────────────────
  DOM.openSettingsBtn.addEventListener('click', () => {
    DOM.apiBaseUrlInput.value = state.apiBaseUrl;
    DOM.apiTestResult.textContent = '';
    DOM.settingsModal.style.display = 'grid';
  });

  DOM.closeSettingsBtn.addEventListener('click', () => {
    DOM.settingsModal.style.display = 'none';
  });

  DOM.testApiBtn.addEventListener('click', async () => {
    let testUrl = normalizeApiUrl(DOM.apiBaseUrlInput.value.trim());
    if (!testUrl) testUrl = PRODUCTION_API_URL;
    DOM.apiBaseUrlInput.value = testUrl;
    DOM.apiTestResult.className = 'api-test-result';
    DOM.apiTestResult.textContent = 'Testing connection...';

    try {
      const res = await fetch(`${testUrl}/api/health`);
      if (res.ok) {
        const data = await res.json();
        DOM.apiTestResult.className = 'api-test-result success';
        DOM.apiTestResult.textContent = `Connected: NexaForge v${data.version} online.`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (e) {
      DOM.apiTestResult.className = 'api-test-result error';
      DOM.apiTestResult.textContent = `Connection failed: ${e.message}`;
    }
  });

  DOM.saveSettingsBtn.addEventListener('click', () => {
    let newUrl = normalizeApiUrl(DOM.apiBaseUrlInput.value.trim());
    if (!newUrl) newUrl = PRODUCTION_API_URL;
    state.apiBaseUrl = newUrl;
    localStorage.setItem('nexaforge_api_url', newUrl);
    showToast('Settings saved: connected to ' + newUrl, 'success');
    DOM.settingsModal.style.display = 'none';
    checkSystemDiagnostics();
  });

  // ── Init & Telemetry Polling ──────────────────────────────────────────────
  checkSystemDiagnostics();
  // Poll telemetry every 30 seconds so visitors and recruiters have live status
  setInterval(() => checkSystemDiagnostics(false), 30000);

  // ── PWA Service Worker Registration ───────────────────────────────────────
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js')
        .then((reg) => {
          console.log('[PWA] Service Worker registered:', reg.scope);
        })
        .catch((err) => {
          console.warn('[PWA] Service Worker registration failed:', err);
        });
    });
  }
})();
