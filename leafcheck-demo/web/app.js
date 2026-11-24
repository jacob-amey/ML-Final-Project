'use strict';

const $ = (selector) => document.querySelector(selector);
const MODELS = [
  { key: 'final177', letter: 'F', name: 'Final Custom CNN', kicker: 'FINAL 177-CLASS MODEL', description: 'Custom architecture · trained from random weights on the full training split', classes: 177 },
  { key: 'cnn', letter: 'A', name: 'Small CNN', kicker: 'ORIGINAL EXPERIMENT', description: 'Original custom CNN · 120 training images', classes: 3 },
  { key: 'starter300', letter: 'B', name: 'Starter 300 CNN', kicker: 'SECOND EXPERIMENT', description: '300-image dataset · 210 used for training', classes: 3 },
  { key: 'createml', letter: 'C', name: 'Apple Create ML', kicker: 'APPLE CLASSIFIER', description: 'Your broader plant image classifier', classes: 177 },
];
const EXAMPLES = [
  { id: 'early_blight', name: 'Early blight', image: 'assets/early_blight.jpg' },
  { id: 'healthy', name: 'Healthy', image: 'assets/healthy.jpg' },
  { id: 'late_blight', name: 'Late blight', image: 'assets/late_blight.jpg' },
];
const state = { models: {}, connected: false, connecting: false, selected: null, objectURL: null, results: null, busy: false, revision: 0, controller: null };
const cards = {};
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const clampScore = (value) => Math.min(1, Math.max(0, Number(value)));
const leafIcon = '<svg aria-hidden="true"><use href="#leaf"/></svg>';

function prettyLabel(raw) {
  const text = String(raw ?? '').replace(/_{2,}/g, ' · ').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
  return text ? text[0].toUpperCase() + text.slice(1) : 'Unknown';
}

function labelParts(raw) {
  const parts = prettyLabel(raw).split(' · ');
  return parts.length > 1
    ? { title: prettyLabel(parts.slice(1).join(' · ')), plant: parts[0] }
    : { title: parts[0], plant: 'Tomato leaf' };
}

function message(text = '', error = false) {
  $('#app-message').hidden = !text;
  $('#app-message').textContent = text;
  $('#app-message').classList.toggle('error', error);
  $('#app-message').setAttribute('role', error ? 'alert' : 'status');
}

function readyCount() {
  return MODELS.filter(({ key }) => state.models[key]?.ready).length;
}

function updateControls() {
  $('#compare-button').disabled = state.busy || !state.selected || !state.connected || readyCount() === 0;
  $('#compare-label').textContent = state.busy ? 'Comparing your leaf…' : 'Run comparison';
  $('#export-button').disabled = state.busy || !state.results;
  for (const { key } of MODELS) cards[key].classList.toggle('loading', state.busy);
}

function placeholder(key, title, text) {
  cards[key].querySelector('.model-result').innerHTML = `<div class="empty-result">${leafIcon}<strong>${escapeHTML(title)}</strong><p>${escapeHTML(text)}</p></div>`;
  cards[key].querySelector('.model-time').textContent = 'Not run yet';
}

function waitingCards() {
  for (const { key } of MODELS) {
    const model = state.models[key];
    if (model && !model.ready) placeholder(key, 'Model unavailable', model.error || 'Check the saved model file and restart the app.');
    else placeholder(key, state.selected ? 'Ready to compare' : 'Waiting for a photo', state.selected ? 'Run the comparison to get a fresh prediction.' : 'Choose a leaf photo to get started.');
  }
}

function summary(heading, text, different = false) {
  $('#comparison-summary').classList.toggle('different', different);
  $('#comparison-summary').innerHTML = `<span class="summary-symbol">${different ? '≠' : 'i'}</span><div><strong>${escapeHTML(heading)}</strong><p>${escapeHTML(text)}</p></div>`;
}

function condition(raw, key) {
  const label = String(raw).toLowerCase().replace(/[^a-z0-9]+/g, ' ');
  if (['createml', 'final177'].includes(key) && !label.includes('tomato')) return null;
  return ['early blight', 'late blight', 'healthy'].find((name) => label.includes(name)) || null;
}

function renderSummary(results) {
  if (results.models.final177?.decision === 'uncertain') {
    summary('LeafCheck withheld its suggestion', 'The other cards show experimental comparisons. Try a clearer photo to get a new LeafCheck result.', true);
    return;
  }
  const available = MODELS.filter(({ key }) => results.models[key]?.label && !results.models[key].error);
  const conditions = available.map(({ key }) => condition(results.models[key].label, key));
  if (available.length < 2) {
    summary('Not enough predictions to compare', 'The model cards explain any unavailable predictions. Check the files and restart the app.');
  } else if (conditions.every(Boolean) && new Set(conditions).size === 1) {
    summary(`${available.length === MODELS.length ? 'All available models agree' : 'The available models agree'}: ${conditions[0]}.`, 'Their top labels describe the same tomato-leaf condition. Agreement is useful context, not proof of accuracy.');
  } else {
    summary('Different answers to explore', 'The two pilots cover three tomato conditions; the final CNN and Create ML cover 177 labels. Different training exposure and label spaces affect their answers.', true);
  }
}

function renderPrediction(key, prediction) {
  if (!prediction || prediction.error) {
    placeholder(key, 'Prediction unavailable', prediction?.error || 'The model did not return a prediction.');
    cards[key].querySelector('.model-time').textContent = 'No result';
    return;
  }
  if (prediction.decision === 'uncertain') {
    placeholder(key, 'Uncertain result', 'LeafCheck withheld its suggestion for this photo. Try a sharp, well-lit close-up of one affected leaf.');
    cards[key].querySelector('.model-time').textContent = `Live comparison · ${Math.round(prediction.elapsed_ms)} ms`;
    return;
  }
  const label = labelParts(prediction.label);
  if (key === 'createml' && !String(prediction.label).includes('__')) label.plant = 'Original model label';
  const entries = Object.entries(prediction.class_scores).sort((a, b) => b[1] - a[1]);
  const score = clampScore(prediction.score);
  cards[key].querySelector('.model-result').innerHTML = `
    <p class="prediction-label">${prediction.decision === 'suggestion' ? 'MODEL SUGGESTION' : 'TOP PREDICTION'}</p>
    <h4 class="prediction-name">${escapeHTML(label.title)}</h4>
    <p class="prediction-plant" title="${escapeHTML(prediction.label)}">${escapeHTML(label.plant)}</p>
    <div class="score-line"><strong class="score-value">${percent(score)}</strong><span>model score</span></div>
    <meter min="0" max="1" value="${score}" aria-label="Top prediction score"></meter>
    <div class="scores"><div class="scores-heading"><span>CLASS SCORES</span><span>${entries.length > 3 ? `TOP 3 OF ${entries.length}` : `ALL ${entries.length} CLASSES`}</span></div>
    ${entries.slice(0, 3).map(([name, value]) => `<div class="score-row"><div><span class="class-label" title="${escapeHTML(name)}">${escapeHTML(prettyLabel(name))}</span><span class="class-value">${percent(value)}</span></div><meter min="0" max="1" value="${clampScore(value)}" aria-label="${escapeHTML(name)} score"></meter></div>`).join('')}</div>`;
  cards[key].querySelector('.model-time').textContent = `Live prediction · ${Math.round(prediction.elapsed_ms)} ms`;
}

function renderDetails() {
  $('#model-details').innerHTML = MODELS.map((info) => {
    const model = state.models[info.key];
    if (!model) return `<section class="detail-section"><h3>${info.name}</h3><p>Connect to the local app to read this saved model’s details.</p></section>`;
    if (!model.ready) return `<section class="detail-section"><h3>${info.name}</h3><p>${escapeHTML(model.error)}</p></section>`;
    const facts = [`${model.class_count} classes`];
    if (model.parameters) facts.push(`${model.parameters.toLocaleString()} parameters`);
    if (model.train_images) facts.push(`${model.train_images} training images`);
    if (model.selected_epoch) facts.push(`Selected epoch ${model.selected_epoch}`);
    const validation = model.deployment_validation || model.validation;
    if (validation) facts.push(`${percent(validation.accuracy)} validation accuracy · ${model.validation_images} images`, `Validation F1 ${validation.macro_f1.toFixed(3)}`);
    const policyNames = { single_view: 'Single image view', d4_probability_mean: 'Average of eight image orientations', train_prior_quarter: 'Training frequency score adjustment' };
    if (policyNames[model.inference_policy]) facts.push(policyNames[model.inference_policy]);
    if (model.test) facts.push(`${percent(model.test.accuracy)} test accuracy · ${model.test_images} images`, `Test F1 ${model.test.macro_f1.toFixed(3)}`);
    const detail = info.key === 'final177'
      ? 'Custom CNN trained from random initialization on all 59,008 training images. Selected by validation macro F1; evaluated once after the model and preprocessing were frozen.'
      : info.key === 'starter300'
      ? 'Second CNN experiment. The 300-image dataset is split into 210 training, 60 validation, and 30 test images.'
      : info.key === 'cnn' ? 'Original pilot CNN. Uses the architecture, class order, and image preparation from your first training notebook.'
      : 'Earlier Apple Create ML classifier. The new test set overlaps its declared old training data in 5,832 images and validation data in 1,683 images. It is not a held-out performance comparison.';
    return `<section class="detail-section"><h3>${info.name}</h3><p>${detail}</p><div class="detail-facts">${facts.map((fact) => `<span>${escapeHTML(fact)}</span>`).join('')}</div><p class="detail-source">Source: ${escapeHTML(model.source)}<br>Image preparation: ${escapeHTML(model.preprocessing)}<br>${escapeHTML(model.comparison_note || 'Historical experiment; its saved metrics use a different evaluation protocol.')}</p></section>`;
  }).join('');
  renderFinalScore();
}

function renderFinalScore() {
  const model = state.models.final177;
  const target = $('#final-score');
  if (!model?.ready || !model.test || !model.baseline) { target.textContent = 'Final evaluation appears here when the frozen model bundle is available.'; return; }
  const score = (name, metric) => `<tr><th scope="row">${name}</th><td>${metric.correct.toLocaleString()} / ${metric.sample_count.toLocaleString()}</td><td>${percent(metric.accuracy)}</td><td>${metric.macro_f1.toFixed(3)}</td></tr>`;
  const supplemental = model.external;
  const field = model.reliability?.field;
  const fieldNote = field ? `<h4>Potato field photo check</h4><p>LeafCheck made suggestions for ${field.accepted} of ${field.support} photos from one field collection. ${field.correct_and_accepted} of those suggestions matched the reference label. Treat suggestions as a starting point for inspection.</p>` : '';
  const supplementalTable = supplemental ? `<h4>Supplementary PlantDoc photos — three tomato labels</h4><table><thead><tr><th>Method</th><th>Correct / total</th><th>Accuracy</th><th>Macro F1 over 3 labels</th></tr></thead><tbody>${score('Final CNN, all 177 outputs eligible', supplemental)}${score('Most common training tomato label', supplemental.conditional_rule)}</tbody></table><p>${supplemental.sample_count.toLocaleString()} photos after overlap screening. This source also appears elsewhere in the master; labels are inherited, with healthy used as a proxy. This is a separate protocol from the 177-class test.</p>` : '';
  target.innerHTML = `<table><thead><tr><th>Method</th><th>Correct / total</th><th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>${score('Final custom CNN', model.test)}${model.reference ? score('Compact177 full-data reference', model.reference) : ''}${score('Most common training label', model.baseline)}</tbody></table><p>Same frozen test set, all 177 classes, scored before any suggestions were withheld. ${model.test.classes_with_fewer_than_10_test_images.toLocaleString()} classes have fewer than ten test images. The Compact177 reference is separate from the earlier three-class pilots. Rule prediction: ${escapeHTML(prettyLabel(model.baseline.rule.class_name))}.</p>${supplementalTable}${fieldNote}`;
}

async function connect() {
  if (state.connecting) return;
  state.connecting = true;
  $('#connection-text').textContent = 'Connecting to local models…';
  $('#retry-button').hidden = true;
  try {
    if (location.protocol === 'file:') throw new Error('Start the app with Launch LeafCheck.command, then use http://127.0.0.1:8765 in your browser.');
    const response = await fetch('/api/status', { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error('The local app could not be reached. Restart the launcher, then retry the connection.');
    const status = await response.json();
    if (!status.app_id?.startsWith('leafcheck-local-')) throw new Error('Restart LeafCheck with the updated launcher, then refresh this page.');
    state.models = status.models;
    await loadFinalExamples();
    state.connected = true;
    const count = readyCount();
    $('#connection-text').textContent = `${count} of ${MODELS.length} models ready on this Mac`;
    $('#connection-status').classList.toggle('connected', count === MODELS.length);
    $('#connection-status').classList.toggle('offline', count !== MODELS.length);
    for (const { key, classes } of MODELS) {
      cards[key].querySelector('.class-count').textContent = `${state.models[key]?.class_count || classes} classes`;
      cards[key].querySelector('.model-ready').textContent = state.models[key]?.ready ? '● Ready' : 'Unavailable';
    }
    message(count === MODELS.length ? '' : 'Some saved models could not load. Their cards show the reason. Available models can still make predictions.', count === 0);
    if (!state.results && !state.busy) waitingCards();
  } catch (error) {
    state.connected = false;
    $('#connection-text').textContent = 'Local app disconnected';
    $('#connection-status').classList.add('offline');
    $('#connection-status').classList.remove('connected');
    $('#retry-button').hidden = false;
    message(error.name === 'TimeoutError' ? 'The local app took too long to respond. Restart the launcher and retry.' : error.message.includes('fetch') || error.name === 'TypeError' ? 'Start Launch LeafCheck.command, then choose Retry connection.' : error.message, true);
  } finally {
    state.connecting = false;
    renderDetails();
    updateControls();
  }
}

function clearObjectURL() {
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.objectURL = null;
}

function invalidateResults() {
  state.revision++;
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
  state.results = null;
  updateControls();
}

function selectImage(selection, url, description) {
  clearObjectURL();
  state.selected = selection;
  state.objectURL = selection.file ? url : null;
  $('#image-preview').src = url;
  $('#image-preview').hidden = false;
  $('#image-preview').alt = selection.example ? `${selection.example.name} plant photograph` : 'Your uploaded leaf photograph';
  $('#drop-zone').classList.add('has-image');
  $('#upload-empty').hidden = true;
  $('#image-badge').hidden = false;
  $('#image-badge').textContent = selection.example ? 'EXAMPLE PHOTO' : 'YOUR PHOTO';
  $('#file-name').textContent = selection.name;
  $('#file-description').textContent = description;
  for (const button of document.querySelectorAll('[data-example]')) button.setAttribute('aria-pressed', String(button.dataset.example === selection.example?.id));
  waitingCards();
  summary('Ready for a fresh comparison', 'Each model will process this photo using the same image preparation as its training or prediction pipeline.');
  $('#results-context').textContent = 'Your photo is ready. Run the comparison.';
  message();
  updateControls();
}

function chooseExample(id) {
  const example = EXAMPLES.find((item) => item.id === id);
  if (!example) return;
  invalidateResults();
  $('#file-input').value = '';
  selectImage({ example, name: example.image.split('/').pop() }, example.image, `${example.description || 'Earlier pilot image; final CNN may have seen it during training'} · Source label: ${prettyLabel(example.actual_class || example.name)}`);
}

async function acceptFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type)) return message('Choose a JPEG or PNG photo.', true);
  if (file.size > 10 * 1024 * 1024) return message('This photo is over 10 MB. Choose a smaller photo.', true);
  const url = URL.createObjectURL(file);
  const revision = state.revision;
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (revision !== state.revision) { URL.revokeObjectURL(url); return; }
    if (image.naturalWidth * image.naturalHeight > 24000000) throw new Error('Choose a photo under 24 megapixels.');
    invalidateResults();
    selectImage({ file, name: file.name }, url, `${image.naturalWidth} × ${image.naturalHeight} pixels · ${Math.round(file.size / 1024)} KB`);
  } catch (error) {
    URL.revokeObjectURL(url);
    message(error.message.startsWith('Choose') ? error.message : 'This image could not be opened. Try another JPEG or PNG.', true);
  }
}

async function compare() {
  if (state.busy || !state.selected || !state.connected || !readyCount()) return;
  const revision = state.revision;
  const selected = state.selected;
  state.busy = true;
  state.results = null;
  state.controller = new AbortController();
  const timer = setTimeout(() => state.controller?.abort(), 90000);
  message();
  updateControls();
  for (const { key } of MODELS) placeholder(key, 'Reading your leaf…', 'Running the saved model on this Mac.');
  $('#results-context').textContent = 'Running your saved models…';
  try {
    const body = selected.file || await (await fetch(selected.example.image)).blob();
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': body.type || 'image/jpeg' }, body, signal: state.controller.signal });
    const results = await response.json();
    if (!response.ok) throw new Error(results.error || 'The comparison could not finish.');
    if (revision !== state.revision) return;
    state.results = { ...results, image_name: selected.name, predicted_at: new Date().toISOString() };
    for (const { key } of MODELS) renderPrediction(key, results.models[key]);
    renderSummary(results);
    $('#results-context').textContent = 'Fresh predictions · Processed on this Mac';
    if (MODELS.some(({ key }) => results.models[key]?.error || !results.models[key])) message('Some models could not finish. Available predictions are shown; the other cards explain the issue.');
  } catch (error) {
    if (revision !== state.revision) return;
    for (const { key } of MODELS) placeholder(key, 'Comparison interrupted', 'Try again, or restart the launcher if the app has stopped.');
    $('#results-context').textContent = 'The comparison did not finish.';
    summary('No complete comparison yet', 'Run the comparison again when the local models are ready.');
    message(error.name === 'AbortError' ? 'The comparison took too long. Restart LeafCheck and try again.' : error.name === 'TypeError' ? 'The local app stopped responding. Restart the launcher and retry the connection.' : error.message, true);
    if (error.name === 'TypeError') {
      state.connected = false;
      $('#connection-text').textContent = 'Local app disconnected';
      $('#connection-status').classList.add('offline');
      $('#retry-button').hidden = false;
    }
  } finally {
    clearTimeout(timer);
    if (revision === state.revision) { state.busy = false; state.controller = null; updateControls(); }
  }
}

function reset() {
  invalidateResults();
  clearObjectURL();
  state.selected = null;
  $('#file-input').value = '';
  $('#image-preview').hidden = true;
  $('#image-preview').removeAttribute('src');
  $('#upload-empty').hidden = false;
  $('#image-badge').hidden = true;
  $('#drop-zone').classList.remove('has-image');
  $('#file-name').textContent = 'No photo selected';
  $('#file-description').textContent = 'A clear photo of a single leaf works best.';
  $('#results-context').textContent = 'Choose a photo, then run the comparison.';
  for (const button of document.querySelectorAll('[data-example]')) button.setAttribute('aria-pressed', 'false');
  waitingCards();
  summary('Read each model in context', 'The final CNN and Create ML cover 177 labels; the earlier pilots cover three tomato conditions. Scores are model outputs, not guaranteed correctness.');
  message();
  updateControls();
}

function exportResults() {
  if (!state.results) return;
  const payload = { ...state.results, model_metadata: state.models,
    note: 'Local model predictions. Scores are not calibrated accuracy. The experiments use different training and evaluation datasets.' };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `leafcheck-${Date.now()}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('#toast').textContent = 'Comparison saved to your downloads.';
  $('#toast').hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3500);
}

function renderExamples() {
  $('#examples').innerHTML = EXAMPLES.map((item) => `<button class="example-button" data-example="${escapeHTML(item.id)}" aria-label="Choose ${escapeHTML(item.name)} example" aria-pressed="false"><img src="${escapeHTML(item.image)}" alt="${escapeHTML(item.name)} plant photograph"><span>${escapeHTML(item.name)}</span></button>`).join('');
}

async function loadFinalExamples() {
  if (!state.models.final177?.ready) return;
  try {
    const response = await fetch('assets/final_examples.json');
    if (!response.ok) return;
    const examples = await response.json();
    if (!Array.isArray(examples) || !examples.length || examples.some((item) => !/^assets\/final_(success|mistake)\.(jpg|jpeg|png)$/i.test(item.image))) return;
    EXAMPLES.splice(0, EXAMPLES.length, ...examples);
    renderExamples();
    $('#example-note').textContent = 'Frozen test examples selected after evaluation: one success and one mistake. They were not used to choose the model.';
  } catch (_) { /* Uploads still work if optional examples are unavailable. */ }
}

function init() {
  for (const info of MODELS) {
    const card = $('#model-card-template').content.firstElementChild.cloneNode(true);
    card.dataset.model = info.key;
    card.querySelector('.model-letter').textContent = info.letter;
    card.querySelector('.model-kicker').textContent = info.kicker;
    card.querySelector('.model-name').textContent = info.name;
    card.querySelector('.model-description').textContent = info.description;
    card.querySelector('.class-count').textContent = `${info.classes} classes`;
    card.querySelector('.details-link').setAttribute('aria-label', `${info.name} details`);
    card.querySelector('.details-link').addEventListener('click', () => $('#details-dialog').showModal());
    cards[info.key] = card;
    $('#model-grid').append(card);
  }
  renderExamples();
  $('#examples').addEventListener('click', (event) => { const button = event.target.closest('[data-example]'); if (button) chooseExample(button.dataset.example); });
  $('#choose-button').addEventListener('click', () => $('#file-input').click());
  $('#drop-zone').addEventListener('click', () => $('#file-input').click());
  $('#drop-zone').addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('#file-input').click(); } });
  $('#file-input').addEventListener('change', (event) => acceptFile(event.target.files[0]));
  for (const type of ['dragenter', 'dragover']) $('#drop-zone').addEventListener(type, (event) => { event.preventDefault(); $('#drop-zone').classList.add('dragging'); });
  $('#drop-zone').addEventListener('dragleave', (event) => { if (!$('#drop-zone').contains(event.relatedTarget)) $('#drop-zone').classList.remove('dragging'); });
  $('#drop-zone').addEventListener('drop', (event) => { event.preventDefault(); $('#drop-zone').classList.remove('dragging'); acceptFile(event.dataTransfer.files[0]); });
  $('#compare-button').addEventListener('click', compare);
  $('#export-button').addEventListener('click', exportResults);
  $('#reset-button').addEventListener('click', reset);
  $('#retry-button').addEventListener('click', connect);
  $('#model-details-button').addEventListener('click', () => $('#details-dialog').showModal());
  for (const id of ['help-button', 'footer-help']) $(`#${id}`).addEventListener('click', () => $('#help-dialog').showModal());
  for (const button of document.querySelectorAll('[data-close]')) button.addEventListener('click', () => button.closest('dialog').close());
  for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  window.addEventListener('beforeunload', clearObjectURL);
  waitingCards();
  connect();
}

init();
