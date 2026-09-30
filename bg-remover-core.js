/* ============================================================
   PRO AI BACKGROUND REMOVER — CORE LOGIC
   ============================================================ */

import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

env.allowLocalModels = false;
env.useBrowserCache = true;

/* ============================================================
   STATE
   ============================================================ */
let currentModelId = 'isnet';
let modelLoaded = false;
let loadedModel = null;
let loadedProcessor = null;

let sourceImage = null;       // HTMLImageElement of original
let sourceCanvas = null;      // Canvas of original
let sourceImageData = null;   // ImageData of original
let alphaMask = null;         // Float32Array of alpha mask (0-1)
let alphaMaskWidth = 0;
let alphaMaskHeight = 0;
let originalAlphaMask = null; // Backup for reset

let editorCanvas = null;
let editorCtx = null;
let currentBg = 'transparent';
let brushMode = 'none';
let isDrawing = false;
let lastBrushPoint = null;

let settings = {
  feather: 0,
  edgeShift: 0,
  threshold: 128,
  alphaMatting: false,
  erodeSize: 10,
  fgThreshold: 240,
  bgThreshold: 10,
  smooth: true,
  despill: true,
  removeShadows: false
};

/* ============================================================
   MODEL CONFIGURATION
   ============================================================ */
const MODELS = {
  isnet: {
    name: 'ISNet General',
    url: 'onnx-community/ISNet-ONNX',
    inputSize: 1024,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  },
  isnet_fp16: {
    name: 'ISNet FP16',
    url: 'onnx-community/ISNet-ONNX',
    inputSize: 1024,
    precision: 'fp16',
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  },
  isnet_quint8: {
    name: 'ISNet Quantized',
    url: 'onnx-community/ISNet-ONNX',
    inputSize: 1024,
    precision: 'q8',
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  },
  u2net_human: {
    name: 'U2Net Human',
    url: 'onnx-community/U2Net-Human-Seg-ONNX',
    inputSize: 320,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  },
  bria_rmbg: {
    name: 'BRIA RMBG 2.0',
    url: 'briaai/RMBG-2.0',
    inputSize: 1024,
    mean: [0.5, 0.5, 0.5],
    std: [1.0, 1.0, 1.0]
  },
  silueta: {
    name: 'Silueta',
    url: 'onnx-community/Silueta-ONNX',
    inputSize: 320,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
  }
};

/* ============================================================
   THEME
   ============================================================ */
window.toggleTheme = function(){
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  document.querySelector('.theme-toggle i').className =
    next === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
};
if(localStorage.getItem('theme') === 'light'){
  document.querySelector('.theme-toggle i').className = 'fas fa-sun';
}

/* ============================================================
   TOAST
   ============================================================ */
let toastTimer;
window.showToast = function(msg, type = 'success'){
  const toast = document.getElementById('toast');
  document.getElementById('toastMsg').textContent = msg;
  toast.className = 'toast show ' + type;
  const icon = toast.querySelector('i');
  icon.className = type === 'error' ? 'fas fa-exclamation-circle'
                 : type === 'info' ? 'fas fa-info-circle'
                 : type === 'warning' ? 'fas fa-exclamation-triangle'
                 : 'fas fa-check-circle';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
};

/* ============================================================
   PROCESSING OVERLAY
   ============================================================ */
function updateProgress(pct){
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = pct + '%';
}
function updateProcessingText(title, msg){
  document.getElementById('processingTitle').textContent = title;
  document.getElementById('processingMsg').textContent = msg;
}
function showProcessing(show, title, msg){
  const overlay = document.getElementById('processingOverlay');
  if(show){
    overlay.classList.add('active');
    if(title) updateProcessingText(title, msg || 'Please wait...');
    updateProgress(0);
  } else {
    overlay.classList.remove('active');
  }
}

/* ============================================================
   MODEL LOADING
   ============================================================ */
window.selectModel = function(modelId, card){
  if(currentModelId === modelId && modelLoaded){
    return;
  }
  currentModelId = modelId;
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
  card.classList.add('active');
  modelLoaded = false;
  loadedModel = null;
  loadedProcessor = null;
  showToast(`Model changed to ${MODELS[modelId].name}`, 'info');
};

async function loadModel(){
  if(modelLoaded && loadedModel && loadedProcessor) {
    return { model: loadedModel, processor: loadedProcessor };
  }

  const config = MODELS[currentModelId];
  showProcessing(true, 'Loading AI Model...', `Downloading ${config.name}...`);

  try {
    // Progress callback for download
    const progressCallback = (progress) => {
      if(progress.status === 'progress' && progress.total){
        const pct = Math.round((progress.loaded / progress.total) * 100);
        updateProgress(pct);
        updateProcessingText('Downloading AI Model...', `${progress.file || 'model'} — ${pct}%`);
      } else if(progress.status === 'done') {
        updateProgress(100);
        updateProcessingText('Model Loaded', 'Initializing...');
      }
    };

    // Load model
    const modelOptions = {
      progress_callback: progressCallback
    };

    // Add precision if specified
    if(config.precision){
      modelOptions.dtype = config.precision;
    }

    loadedModel = await AutoModel.from_pretrained(config.url, modelOptions);
    loadedProcessor = await AutoProcessor.from_pretrained(config.url);
    modelLoaded = true;

    showProcessing(false);
    showToast('✅ AI Model loaded successfully!');
    return { model: loadedModel, processor: loadedProcessor };
  } catch(err){
    console.error('Model load error:', err);
    showProcessing(false);
    showToast('Model load failed: ' + err.message, 'error');
    throw err;
  }
}

/* ============================================================
   IMAGE PREPROCESSING
   ============================================================ */
async function preprocessImage(image, targetSize, mean, std){
  // Create canvas at target size
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d');

  // Calculate aspect-preserving resize
  const scale = Math.min(targetSize / image.width, targetSize / image.height);
  const w = Math.round(image.width * scale);
  const h = Math.round(image.height * scale);
  const x = Math.round((targetSize - w) / 2);
  const y = Math.round((targetSize - h) / 2);

  // Fill with grey (ImageNet pad color)
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, targetSize, targetSize);

  // Draw resized image
  ctx.drawImage(image, x, y, w, h);

  // Extract pixel data
  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const data = imageData.data;

  // Normalize: (pixel/255 - mean) / std, CHW format
  const float32Data = new Float32Array(3 * targetSize * targetSize);
  const pixelCount = targetSize * targetSize;

  for(let i = 0; i < pixelCount; i++){
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    float32Data[i] = (r - mean[0]) / std[0];
    float32Data[pixelCount + i] = (g - mean[1]) / std[1];
    float32Data[pixelCount * 2 + i] = (b - mean[2]) / std[2];
  }

  return {
    tensor: float32Data,
    dims: [1, 3, targetSize, targetSize],
    originalCoords: { x, y, w, h }
  };
}

/* ============================================================
   RUN INFERENCE
   ============================================================ */
async function runInference(image){
  const config = MODELS[currentModelId];
  const { model, processor } = await loadModel();

  showProcessing(true, 'Removing Background...', 'AI is analyzing your image...');
  updateProgress(30);

  try {
    // Preprocess
    const preprocessed = await preprocessImage(
      image,
      config.inputSize,
      config.mean,
      config.std
    );

    updateProgress(50);

    // Create input tensor
    const inputTensor = new (await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0')).Tensor(
      'float32',
      preprocessed.tensor,
      preprocessed.dims
    );

    // Run model
    const output = await model({ 'input': inputTensor });

    updateProgress(75);

    // Extract mask from output
    let maskTensor;
    // Different models output different keys
    const outputKeys = Object.keys(output);
    for(const key of outputKeys){
      const tensor = output[key];
      if(tensor && tensor.dims && tensor.dims.length === 4){
        // [1, 1, H, W] - mask output
        if(tensor.dims[1] === 1){
          maskTensor = tensor;
          break;
        }
      }
    }
    if(!maskTensor && outputKeys.length > 0){
      maskTensor = output[outputKeys[outputKeys.length - 1]];
    }

    if(!maskTensor){
      throw new Error('No mask output from model');
    }

    // Get mask dims and data
    const maskDims = maskTensor.dims;
    const maskH = maskDims[2];
    const maskW = maskDims[3];
    const maskData = maskTensor.data;

    // Normalize mask to 0-1 range
    let minVal = Infinity, maxVal = -Infinity;
    for(let i = 0; i < maskData.length; i++){
      if(maskData[i] < minVal) minVal = maskData[i];
      if(maskData[i] > maxVal) maxVal = maskData[i];
    }
    const range = maxVal - minVal || 1;

    // Store alpha mask
    alphaMaskWidth = maskW;
    alphaMaskHeight = maskH;
    alphaMask = new Float32Array(maskW * maskH);
    for(let i = 0; i < maskW * maskH; i++){
      alphaMask[i] = (maskData[i] - minVal) / range;
    }

    // Backup for reset
    originalAlphaMask = new Float32Array(alphaMask);

    updateProgress(100);
    return true;
  } catch(err){
    console.error('Inference error:', err);
    showProcessing(false);
    showToast('Inference failed: ' + err.message, 'error');
    throw err;
  }
}

/* ============================================================
   RENDER RESULT TO CANVAS
   ============================================================ */
function renderResult(){
  if(!sourceImage || !alphaMask) return;

  const W = sourceImage.naturalWidth;
  const H = sourceImage.naturalHeight;

  // Create temp canvas for mask resize
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = alphaMaskWidth;
  maskCanvas.height = alphaMaskHeight;
  const maskCtx = maskCanvas.getContext('2d');
  const maskImageData = maskCtx.createImageData(alphaMaskWidth, alphaMaskHeight);

  // Apply threshold & edge refinement to alpha mask
  const feather = settings.feather;
  const edgeShift = settings.edgeShift;
  const threshold = settings.threshold / 255;

  for(let i = 0; i < alphaMask.length; i++){
    let a = alphaMask[i];

    // Edge shift: expand or contract the mask
    if(edgeShift !== 0){
      a = Math.pow(a, 1 + edgeShift * 0.1);
    }

    // Feather: smooth transition
    if(feather > 0){
      a = smoothstep(a - feather * 0.02, a + feather * 0.02, a);
    }

    // Threshold: binarize
    if(threshold > 0 && threshold < 1){
      a = a < threshold ? Math.max(0, a - 0.1) : Math.min(1, a + 0.1);
    }

    a = Math.max(0, Math.min(1, a));

    maskImageData.data[i * 4] = 255;
    maskImageData.data[i * 4 + 1] = 255;
    maskImageData.data[i * 4 + 2] = 255;
    maskImageData.data[i * 4 + 3] = Math.round(a * 255);
  }

  maskCtx.putImageData(maskImageData, 0, 0);

  // Resize mask to source size
  const resizedMask = document.createElement('canvas');
  resizedMask.width = W;
  resizedMask.height = H;
  const rCtx = resizedMask.getContext('2d');
  rCtx.imageSmoothingEnabled = true;
  rCtx.imageSmoothingQuality = 'high';
  rCtx.drawImage(maskCanvas, 0, 0, W, H);

  const resizedMaskData = rCtx.getImageData(0, 0, W, H);

  // Create source canvas
  if(!sourceCanvas){
    sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = W;
    sourceCanvas.height = H;
    sourceCanvas.getContext('2d').drawImage(sourceImage, 0, 0);
  }
  const sCtx = sourceCanvas.getContext('2d');
  const srcData = sCtx.getImageData(0, 0, W, H);

  // Apply mask to source
  const resultData = new ImageData(W, H);
  for(let i = 0; i < W * H; i++){
    resultData.data[i * 4] = srcData.data[i * 4];
    resultData.data[i * 4 + 1] = srcData.data[i * 4 + 1];
    resultData.data[i * 4 + 2] = srcData.data[i * 4 + 2];
    resultData.data[i * 4 + 3] = resizedMaskData.data[i * 4 + 3];
  }

  // Post-processing: despill
  if(settings.despill){
    despillGreen(resultData);
  }

  // Draw to editor canvas
  if(!editorCanvas){
    editorCanvas = document.getElementById('editCanvas');
  }
  editorCanvas.width = W;
  editorCanvas.height = H;
  editorCtx = editorCanvas.getContext('2d');
  editorCtx.putImageData(resultData, 0, 0);

  // Apply background
  applyBackground();
}

function smoothstep(edge0, edge1, x){
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0 || 1)));
  return t * t * (3 - 2 * t);
}

function despillGreen(imageData){
  const data = imageData.data;
  for(let i = 0; i < data.length; i += 4){
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    // If green is dominant and alpha is partial (edge region)
    if(data[i + 3] < 255 && data[i + 3] > 0){
      const maxRB = Math.max(r, b);
      if(g > maxRB){
        data[i + 1] = maxRB;
      }
    }
  }
}

/* ============================================================
   BACKGROUND APPLICATION
   ============================================================ */
window.setBg = function(bg, swatch){
  currentBg = bg;
  document.querySelectorAll('.bg-swatch').forEach(s => s.classList.remove('active'));
  if(swatch) swatch.classList.add('active');
  applyBackground();
};

function applyBackground(){
  const wrapper = document.getElementById('canvasWrapper');
  if(!wrapper) return;

  if(currentBg === 'transparent'){
    wrapper.style.background = '';
    wrapper.style.backgroundImage = '';
    wrapper.style.backgroundColor = '';
    wrapper.classList.remove('no-checker');
  } else if(currentBg === 'gradient'){
    wrapper.classList.add('no-checker');
    wrapper.style.background = 'linear-gradient(135deg,#6366f1,#ec4899)';
    wrapper.style.backgroundImage = '';
    wrapper.style.backgroundColor = '';
  } else {
    wrapper.classList.add('no-checker');
    wrapper.style.background = currentBg;
    wrapper.style.backgroundImage = 'none';
    wrapper.style.backgroundColor = currentBg;
  }
}

/* ============================================================
   BRUSH MODES
   ============================================================ */
window.setBrushMode = function(mode, btn){
  brushMode = mode;
  document.querySelectorAll('.brush-btn').forEach(b => b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  const wrapper = document.getElementById('canvasWrapper');
  if(wrapper){
    if(mode === 'none') wrapper.classList.add('brush-none');
    else wrapper.classList.remove('brush-none');
  }
};

function getCanvasCoords(e){
  const rect = editorCanvas.getBoundingClientRect();
  const scaleX = editorCanvas.width / rect.width;
  const scaleY = editorCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function applyBrushAt(x, y, size, mode){
  if(!alphaMask || !editorCanvas) return;

  const W = editorCanvas.width;
  const H = editorCanvas.height;
  const radius = size / 2;

  // Convert to mask coordinates
  const maskScaleX = alphaMaskWidth / W;
  const maskScaleY = alphaMaskHeight / H;

  const mx = x * maskScaleX;
  const my = y * maskScaleY;
  const mr = radius * Math.max(maskScaleX, maskScaleY);

  // Iterate over mask pixels in brush area
  const startX = Math.max(0, Math.floor(mx - mr));
  const endX = Math.min(alphaMaskWidth, Math.ceil(mx + mr));
  const startY = Math.max(0, Math.floor(my - mr));
  const endY = Math.min(alphaMaskHeight, Math.ceil(my + mr));

  for(let py = startY; py < endY; py++){
    for(let px = startX; px < endX; px++){
      const dx = px - mx;
      const dy = py - my;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if(dist <= mr){
        const idx = py * alphaMaskWidth + px;
        // Soft falloff
        const falloff = 1 - (dist / mr);
        if(mode === 'erase'){
          alphaMask[idx] = Math.max(0, alphaMask[idx] - falloff);
        } else if(mode === 'restore'){
          alphaMask[idx] = Math.min(1, alphaMask[idx] + falloff);
        }
      }
    }
  }

  // Re-render
  renderResult();
}

/* ============================================================
   CANVAS MOUSE EVENTS
   ============================================================ */
function initCanvasEvents(){
  if(!editorCanvas) return;

  editorCanvas.addEventListener('mousedown', (e) => {
    if(brushMode === 'none') return;
    isDrawing = true;
    const coords = getCanvasCoords(e);
    const size = parseInt(document.getElementById('brushSize').value);
    applyBrushAt(coords.x, coords.y, size, brushMode);
    lastBrushPoint = coords;
  });

  editorCanvas.addEventListener('mousemove', (e) => {
    if(!isDrawing || brushMode === 'none') return;
    const coords = getCanvasCoords(e);
    const size = parseInt(document.getElementById('brushSize').value);
    applyBrushAt(coords.x, coords.y, size, brushMode);
    lastBrushPoint = coords;
  });

  editorCanvas.addEventListener('mouseup', () => {
    isDrawing = false;
    lastBrushPoint = null;
  });

  editorCanvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    lastBrushPoint = null;
  });

  // Touch support
  editorCanvas.addEventListener('touchstart', (e) => {
    if(brushMode === 'none') return;
    e.preventDefault();
    isDrawing = true;
    const touch = e.touches[0];
    const coords = getCanvasCoords(touch);
    const size = parseInt(document.getElementById('brushSize').value);
    applyBrushAt(coords.x, coords.y, size, brushMode);
  }, { passive: false });

  editorCanvas.addEventListener('touchmove', (e) => {
    if(!isDrawing || brushMode === 'none') return;
    e.preventDefault();
    const touch = e.touches[0];
    const coords = getCanvasCoords(touch);
    const size = parseInt(document.getElementById('brushSize').value);
    applyBrushAt(coords.x, coords.y, size, brushMode);
  }, { passive: false });

  editorCanvas.addEventListener('touchend', () => {
    isDrawing = false;
  });
}

/* ============================================================
   SETTINGS CONTROLS
   ============================================================ */
window.applyEdgeRefinement = function(){
  settings.feather = parseFloat(document.getElementById('featherAmount').value);
  settings.edgeShift = parseFloat(document.getElementById('edgeShift').value);
  if(alphaMask) renderResult();
};

window.applyThreshold = function(){
  settings.threshold = parseInt(document.getElementById('alphaThreshold').value);
  if(alphaMask) renderResult();
};

window.toggleAlphaMatting = function(){
  settings.alphaMatting = !settings.alphaMatting;
  const toggle = document.getElementById('alphaMattingToggle');
  toggle.classList.toggle('active', settings.alphaMatting);
  showToast(settings.alphaMatting ? 'Alpha Matting enabled (experimental)' : 'Alpha Matting disabled', 'info');
};

window.toggleSmooth = function(){
  settings.smooth = !settings.smooth;
  document.getElementById('smoothToggle').classList.toggle('active', settings.smooth);
  if(alphaMask) renderResult();
};

window.toggleDespill = function(){
  settings.despill = !settings.despill;
  document.getElementById('despillToggle').classList.toggle('active', settings.despill);
  if(alphaMask) renderResult();
};

window.toggleShadow = function(){
  settings.removeShadows = !settings.removeShadows;
  document.getElementById('shadowToggle').classList.toggle('active', settings.removeShadows);
  if(alphaMask) renderResult();
};

/* ============================================================
   FILE HANDLING
   ============================================================ */
window.processFile = async function(file){
  if(!file.type.startsWith('image/')){
    showToast('Please select an image file', 'error');
    return;
  }
  if(file.size > 30 * 1024 * 1024){
    showToast('Image too large (max 30MB)', 'error');
    return;
  }

  // Read as data URL
  const dataURL = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  // Load image
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = dataURL;
  });

  // Store
  sourceImage = img;
  sourceCanvas = null;
  sourceImageData = null;

  // Show result section
  document.getElementById('uploadSection').style.display = 'none';
  document.getElementById('resultSection').classList.add('active');

  // Run inference
  try {
    await runInference(img);
    showProcessing(false);
    renderResult();
    initCanvasEvents();
    showToast('✅ Background removed! Use brushes to refine.');
  } catch(err){
    showProcessing(false);
    showToast('Processing failed: ' + err.message, 'error');
  }
};

/* ============================================================
   DRAG & DROP
   ============================================================ */
const dropzone = document.getElementById('dropzone');
['dragenter','dragover'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if(file && file.type.startsWith('image/')) processFile(file);
  else showToast('Please drop an image file', 'error');
});
dropzone.addEventListener('click', e => {
  if(e.target.tagName !== 'BUTTON' && !e.target.closest('button')){
    document.getElementById('fileInput').click();
  }
});

document.getElementById('fileInput').addEventListener('change', e => {
  const file = e.target.files[0];
  if(file) processFile(file);
  e.target.value = '';
});

/* ============================================================
   CLIPBOARD PASTE
   ============================================================ */
document.addEventListener('paste', e => {
  const items = e.clipboardData?.items;
  if(!items) return;
  for(let item of items){
    if(item.type.startsWith('image/')){
      const file = item.getAsFile();
      if(file){
        processFile(file);
        showToast('Image pasted!');
        break;
      }
    }
  }
});

window.pasteFromClipboard = async function(){
  try {
    if(!navigator.clipboard || !navigator.clipboard.read){
      showToast('Use Ctrl+V to paste', 'info');
      return;
    }
    const items = await navigator.clipboard.read();
    for(const item of items){
      for(const type of item.types){
        if(type.startsWith('image/')){
          const blob = await item.getType(type);
          processFile(new File([blob], 'pasted.png', { type }));
          return;
        }
      }
    }
    showToast('No image in clipboard', 'error');
  } catch(err){
    showToast('Press Ctrl+V to paste', 'info');
  }
};

/* ============================================================
   LOAD FROM URL
   ============================================================ */
window.loadFromURL = async function(){
  const url = document.getElementById('urlInput').value.trim();
  if(!url){
    showToast('Please enter a URL', 'warning');
    return;
  }
  try {
    showToast('Loading image...', 'info');
    const response = await fetch(url, { mode: 'cors' });
    const blob = await response.blob();
    if(!blob.type.startsWith('image/')) throw new Error('Not an image');
    processFile(new File([blob], 'from-url.jpg', { type: blob.type }));
  } catch(err){
    showToast('Failed to load image (CORS may be blocked)', 'error');
  }
};

/* ============================================================
   DOWNLOAD / COPY
   ============================================================ */
async function generateFinalImage(format = 'png'){
  if(!editorCanvas) throw new Error('No image to process');

  const canvas = document.createElement('canvas');
  canvas.width = editorCanvas.width;
  canvas.height = editorCanvas.height;
  const ctx = canvas.getContext('2d');

  // Fill background if needed
  if(format !== 'png' || currentBg !== 'transparent'){
    if(currentBg === 'gradient'){
      const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      grad.addColorStop(0, '#6366f1');
      grad.addColorStop(1, '#ec4899');
      ctx.fillStyle = grad;
    } else if(currentBg !== 'transparent'){
      ctx.fillStyle = currentBg;
    } else {
      ctx.fillStyle = '#ffffff';
    }
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  ctx.drawImage(editorCanvas, 0, 0);

  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => {
      if(blob) resolve(blob);
      else reject(new Error('Failed to create blob'));
    }, 'image/' + format, 0.95);
  });
}

window.downloadResult = async function(format){
  try {
    showToast('Preparing download...', 'info');
    const blob = await generateFinalImage(format);
    const ext = format === 'jpeg' ? 'jpg' : format;
    const filename = `bg-removed-${Date.now()}.${ext}`;
    downloadBlob(blob, filename);
    showToast('✅ Downloaded: ' + filename);
  } catch(err){
    console.error(err);
    showToast('Download failed: ' + err.message, 'error');
  }
};

window.copyToClipboard = async function(){
  try {
    const blob = await generateFinalImage('png');
    if(navigator.clipboard && window.ClipboardItem){
      await navigator.clipboard.write([
        new ClipboardItem({ 'image/png': blob })
      ]);
      showToast('✅ Copied to clipboard!');
    } else {
      showToast('Clipboard not supported — download instead', 'error');
    }
  } catch(err){
    showToast('Copy failed: ' + err.message, 'error');
  }
};

function downloadBlob(blob, filename){
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

/* ============================================================
   RESET
   ============================================================ */
window.resetAll = function(){
  sourceImage = null;
  sourceCanvas = null;
  sourceImageData = null;
  alphaMask = null;
  originalAlphaMask = null;
  currentBg = 'transparent';

  document.getElementById('uploadSection').style.display = 'block';
  document.getElementById('resultSection').classList.remove('active');
  document.getElementById('urlInput').value = '';

  document.querySelectorAll('.bg-swatch').forEach(s => s.classList.remove('active'));
  const tSwatch = document.querySelector('.bg-swatch.transparent');
  if(tSwatch) tSwatch.classList.add('active');

  if(editorCanvas){
    const ctx = editorCanvas.getContext('2d');
    ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
};

/* ============================================================
   MODE SWITCHING
   ============================================================ */
window.switchMode = function(mode, btn){
  document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('singleMode').style.display = mode === 'single' ? 'block' : 'none';
  document.getElementById('batchMode').style.display = mode === 'batch' ? 'block' : 'none';
};

/* ============================================================
   BATCH PROCESSING
   ============================================================ */
let batchItems = [];
let batchResults = [];

window.processBatchFiles = async function(files){
  if(files.length > 10){
    showToast('Max 10 images per batch', 'warning');
    files = files.slice(0, 10);
  }

  batchItems = files.map(f => ({ file: f, blob: null, status: 'pending' }));
  batchResults = [];

  const grid = document.getElementById('batchGrid');
  document.getElementById('batchResults').style.display = 'block';
  grid.innerHTML = '';

  // Render placeholders
  for(let i = 0; i < files.length; i++){
    const item = document.createElement('div');
    item.className = 'batch-item';
    item.id = 'batch-item-' + i;
    const url = URL.createObjectURL(files[i]);
    item.innerHTML = `
      <img class="batch-thumb" src="${url}" alt="Image ${i+1}">
      <button class="batch-download" onclick="downloadBatchItem(${i})" title="Download">
        <i class="fas fa-download"></i>
      </button>
      <div class="batch-info">
        <div class="batch-name">${files[i].name}</div>
        <div class="batch-status pending"><i class="fas fa-clock"></i> Pending</div>
      </div>
    `;
    grid.appendChild(item);
  }

  showProcessing(true, 'Batch Processing...', `0 of ${files.length} done`);

  for(let i = 0; i < files.length; i++){
    const item = document.getElementById('batch-item-' + i);
    const statusEl = item.querySelector('.batch-status');
    statusEl.className = 'batch-status processing';
    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    updateProcessingText(`Batch ${i+1} of ${files.length}`, `Processing: ${files[i].name}`);
    updateProgress(0);

    try {
      // Load image
      const img = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = reject;
          image.src = e.target.result;
        };
        reader.readAsDataURL(files[i]);
      });

      // Store as source temporarily
      const prevImage = sourceImage;
      const prevCanvas = sourceCanvas;
      const prevMask = alphaMask;

      sourceImage = img;
      sourceCanvas = null;

      // Run inference
      await runInference(img);

      // Create result canvas
      const resultCanvas = document.createElement('canvas');
      resultCanvas.width = img.naturalWidth;
      resultCanvas.height = img.naturalHeight;
      const rCtx = resultCanvas.getContext('2d');

      // Render
      const tempEditorCanvas = editorCanvas;
      const tempEditorCtx = editorCtx;
      editorCanvas = resultCanvas;
      editorCtx = rCtx;
      renderResult();

      // Get blob
      const blob = await new Promise(res => resultCanvas.toBlob(res, 'image/png'));

      // Restore
      editorCanvas = tempEditorCanvas;
      editorCtx = tempEditorCtx;
      sourceImage = prevImage;
      sourceCanvas = prevCanvas;
      alphaMask = prevMask;

      batchItems[i].blob = blob;
      batchItems[i].status = 'done';

      const resultURL = URL.createObjectURL(blob);
      item.classList.add('done');
      item.querySelector('.batch-thumb').src = resultURL;
      statusEl.className = 'batch-status done';
      statusEl.innerHTML = '<i class="fas fa-check-circle"></i> Done';

      batchResults.push({ name: files[i].name, blob });
    } catch(err){
      console.error(err);
      statusEl.className = 'batch-status error';
      statusEl.innerHTML = '<i class="fas fa-times-circle"></i> Failed';
      batchItems[i].status = 'error';
    }
  }

  showProcessing(false);
  showToast(`✅ Batch complete: ${batchResults.length} of ${files.length}`);
};

window.downloadBatchItem = function(index){
  const item = batchItems[index];
  if(!item || !item.blob) return;
  const filename = 'bg-removed-' + (item.file.name.replace(/\.[^.]+$/, '') || index) + '.png';
  downloadBlob(item.blob, filename);
  showToast('Downloaded: ' + filename);
};

window.downloadAllBatch = async function(){
  if(batchResults.length === 0){
    showToast('No results to download', 'error');
    return;
  }
  for(let i = 0; i < batchResults.length; i++){
    const r = batchResults[i];
    const filename = 'bg-removed-' + (r.name.replace(/\.[^.]+$/, '') || i) + '.png';
    downloadBlob(r.blob, filename);
    await new Promise(res => setTimeout(res, 300));
  }
  showToast(`✅ Downloaded ${batchResults.length} files`);
};

window.clearBatch = function(){
  batchItems = [];
  batchResults = [];
  document.getElementById('batchResults').style.display = 'none';
  document.getElementById('batchGrid').innerHTML = '';
};

/* ============================================================
   BATCH DROPZONE
   ============================================================ */
const batchDropzone = document.getElementById('batchDropzone');
['dragenter','dragover'].forEach(evt => {
  batchDropzone.addEventListener(evt, e => {
    e.preventDefault();
    batchDropzone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(evt => {
  batchDropzone.addEventListener(evt, e => {
    e.preventDefault();
    batchDropzone.classList.remove('dragover');
  });
});
batchDropzone.addEventListener('drop', e => {
  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
  if(files.length) processBatchFiles(files);
  else showToast('Please drop image files', 'error');
});
batchDropzone.addEventListener('click', e => {
  if(e.target.tagName !== 'BUTTON' && !e.target.closest('button')){
    document.getElementById('batchFileInput').click();
  }
});

document.getElementById('batchFileInput').addEventListener('change', e => {
  const files = Array.from(e.target.files);
  if(files.length) processBatchFiles(files);
  e.target.value = '';
});

/* ============================================================
   INIT
   ============================================================ */
window.addEventListener('DOMContentLoaded', () => {
  showToast('Ready! Select a model and upload an image.', 'info');
});

window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
