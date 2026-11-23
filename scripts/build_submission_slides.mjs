// Build the local defense deck from frozen experiment and demo records.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [root, buildDir, finalPath] = process.argv.slice(2).map(value => path.resolve(value));
const { RUNTIME_NODE_MODULES, RUNTIME_PYTHON, PRESENTATION_SKILL_DIR } = process.env;
const requireFromRuntime = createRequire(path.join(RUNTIME_NODE_MODULES, 'leafcheck.cjs'));
const { Presentation, PresentationFile } = await import(pathToFileURL(requireFromRuntime.resolve('@oai/artifact-tool')).href);
const toolsDir = path.join(PRESENTATION_SKILL_DIR, 'container_tools');
const { finalizePresentation } = await import(pathToFileURL(path.join(toolsDir, 'artifact_tool_utils.mjs')).href);
const bundle = path.join(root, 'artifacts/leafcheck177-v1');
const docs = path.join(root, 'docs');
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const [freeze, config, manifest, final, compact, rule, supplement, policies, duration, browser, examples] = await Promise.all([
  'freeze.json', 'config.json', 'artifact.json', 'test_metrics.json', 'reference/test_metrics.json',
  'baseline_metrics.json', 'external_metrics.json', 'inference_policy_selection.json', 'docs/duration_trial_decision.json',
  'docs/demo_verification.json', 'demo_examples.json',
].map(file => read(file.startsWith('docs/') ? path.join(root, file) : path.join(bundle, file))));
if (!manifest.final_evaluation_complete || !manifest.reference_evaluation_complete || !manifest.supplementary_evaluation_complete ||
    browser.checkpoint_sha256 !== freeze.checkpoint_sha256) throw Error('Complete the frozen model and browser verification first');

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = '#172B23', green = '#17523E', muted = '#4F6258', paper = '#F6F8F3';
const percentage = value => `${(100 * value).toFixed(2)}%`;
const f1 = value => value.toFixed(6);
const number = value => value.toLocaleString('en-US');
const pretty = label => label.replaceAll('__', ' / ').replaceAll('_', ' ');
const scoreNames = { single_view: 'Single view', d4_probability_mean: 'Eight orientation scores averaged', train_prior_quarter: 'Training-frequency adjustment' };

function text(slide, content, left, top, width, height, size, opts = {}) {
  const shape = slide.shapes.add({
    name: opts.name || 'Editable text', geometry: 'textbox', position: { left, top, width, height },
    fill: 'none', line: { fill: 'none', width: 0 },
  });
  shape.text = content;
  shape.text.style = { fontFamily: 'Arial', fontSize: `${size}pt`, color: opts.color || ink,
    bold: opts.bold || false, alignment: opts.align || 'left', verticalAlignment: 'middle',
    insets: { top: 0, left: 0, bottom: 0, right: 0 } };
  return shape;
}

function slide(title, notes) {
  const page = deck.slides.add();
  page.background.fill = paper;
  if (title) text(page, title, 64, 64, 1152, 76, 33, { bold: true, name: 'Slide title' });
  page.speakerNotes.text = notes;
  return page;
}

async function image(slide, file, left, top, width, height, alt, fit = 'contain') {
  const bytes = await fs.readFile(file);
  slide.images.add({ blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: file.endsWith('.png') ? 'image/png' : 'image/jpeg', alt, fit,
    position: { left, top, width, height } });
}

function lines(slide, rows, { left = 66, top = 170, width = 1110, step = 100, textSize = 23 } = {}) {
  for (let i = 0; i < rows.length; i++) {
    text(slide, rows[i][0], left, top + i * step, width, 34, 19, { bold: true, color: green });
    text(slide, rows[i][1], left, top + i * step + 37, width, step - 42, textSize);
  }
}

const cover = slide(null, 'Introduce the CSC-427 team and describe the single-photo classification task. The cover photo is an existing LeafCheck example and is illustrative.');
text(cover, 'CSC-427  FINAL PROJECT', 66, 116, 680, 50, 17, { bold: true, color: green });
text(cover, 'Plant Health Classification\nand an Interactive Application', 66, 180, 715, 210, 42, { bold: true, name: 'Presentation title' });
text(cover, 'LeafCheck', 66, 435, 590, 70, 28, { color: green });
await image(cover, path.join(root, 'leafcheck-demo/web/assets/healthy.jpg'), 802, 0, 478, 720, 'Illustrative plant photo from the LeafCheck project', 'cover');

const dataset = slide('Dataset and splits', 'Explain that a prediction is correct when its top label matches the inherited reference annotation. Discuss exact duplicates, near-duplicate grouping and missing plant IDs. Sources: docs/final_dataset_audit.md and data/README.md.');
for (const [i, [count, label]] of [['59,008','Training'], ['16,836','Validation'], ['8,425','Reserved test']].entries()) {
  const left = 66 + i * 396;
  text(dataset, count, left, 182, 367, 92, 49, { bold: true, color: green });
  text(dataset, label, left, 281, 367, 50, 22);
}
lines(dataset, [['177 inherited labels','Crop or species paired with a plant condition.'],
  ['Related images stay together','The audit uses source records, known leaf IDs and duplicate checks.']], { top: 376, step: 102, textSize: 21 });

const architecture = slide('Custom residual CNN', 'Each fresh run uses PyTorch initialization without pretrained weights. Explain the skip connection and how global pooling reduces the classifier size. Residual learning reference: He et al., CVPR 2016, doi:10.1109/CVPR.2016.90.');
lines(architecture, [['Image preparation', `RGB at ${config.input_size} × ${config.input_size}, aspect ratio preserved, training-only normalization.`],
  ['Feature extractor', `Stride-2 stem and four residual blocks with ${config.model.channels.join(', ')} channels.`],
  ['Classifier', `Global average pooling, dropout during training, then 177 raw class scores.`],
  ['Parameters learned from the training split', `${number(freeze.parameters)} trainable parameters. Epoch ${freeze.selected_epoch} supplies the delivered weights.`]],
  { step: 107, textSize: 21 });

const selection = slide('Validation decisions', 'Scores are macro-F1 averaged across 177 labels. Compare two seeds under the same setup. The project retained mild illumination and lighter decay. Higher resolution was accepted only by the declared paired rule. See docs/experiments.csv and docs/duration_trial_decision.json. Primary scoring-policy sources: https://arxiv.org/abs/2007.07314 and https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html.');
const sequence = [['128 px weighted','0.600999'], ['160 px with illumination','0.621631'], ['224 px with illumination','0.641060']];
sequence.push([
  duration.native_size_accepted ? '256 px with illumination' : '256 px trial',
  f1((await read(path.join(docs, 'seed_stability_256.json'))).mean_macro_f1),
]);
for (const [i, [label, value]] of sequence.entries()) {
  text(selection, label, 66, 170 + i * 66, 580, 57, 22);
  text(selection, value, 655, 170 + i * 66, 250, 57, 25, { bold: true, color: green });
}
text(selection, duration.native_size_accepted
  ? 'Two-seed mean validation macro-F1'
  : 'The 256 px gain fell below the declared 0.005 threshold.', 67, 456, 1130, 48, 18, { color: muted });
text(selection, `${scoreNames[freeze.inference_policy.name]}\nDelivered validation macro-F1 ${f1(policies.deployment_validation.macro_f1)}`, 67, 539, 1140, 105, 26, { bold: true });

const heldout = slide('Reserved test results', 'All three methods use exactly the same 8,425 images. The compact CNN trained on the full training split. The rule predicts the most frequent training class. The live Create ML adapter has overlapping declared training exposure. The earlier CNNs use three classes.');
const measures = [['Method','Correct / total','Accuracy','Macro-F1'], ['Selected CNN',`${number(final.correct)} / 8,425`,percentage(final.accuracy),f1(final.macro_f1)],
  ['Compact177 reference',`${number(compact.correct)} / 8,425`,percentage(compact.accuracy),f1(compact.macro_f1)],
  ['Most common training class',`${number(rule.correct)} / 8,425`,percentage(rule.accuracy),f1(rule.macro_f1)]];
const resultsTable = heldout.tables.add({ rows: measures.length, columns: 4, left: 65, top: 181, width: 1150, height: 275,
  values: measures, columnWidths: [452, 232, 222, 244] });
resultsTable.cells.block({ row: 0, column: 0, rowCount: measures.length, columnCount: 4 }).textStyle.fontSize = 26;
for (let i = 0; i < measures.length; i++) {
  resultsTable.rows[i].height = 68;
  for (let j = 0; j < 4; j++) resultsTable.getCell(i,j).fill = i === 0 ? '#DDEAE0' : i % 2 === 0 ? '#EDF3ED' : '#FFFFFF';
}
text(heldout, `${final.classes_with_fewer_than_10_test_images} classes have fewer than ten test images.`, 66, 513, 1150, 52, 21, { color: muted });
text(heldout, `PlantDoc supplement: ${supplement.correct} of 183 correct across three reference labels.`, 66, 573, 1150, 52, 21, { color: muted });

const demo = slide('Live application demo', 'Open the local LeafCheck launcher. Upload a real file and run comparison. Explain that the browser sends photo bytes to the same Python server that hosts the page. Python returns structured model scores. Point out that scores are uncalibrated. Show the fair held-out score table.');
lines(demo, [['Upload', 'Choose a JPEG or PNG photo and run the comparison.'], ['Predict', 'The local Python server applies the saved preprocessing and CNN.'],
  ['Display', 'The browser shows the top label and its score beside the other models.']], { top: 181, step: 122, textSize: 24 });
text(demo, '127.0.0.1:8765', 66, 584, 1150, 55, 25, { color: green, bold: true });

const samples = slide('One success and one mistake', 'Use the exact saved held-out examples. Show the reference annotation and top prediction. Explain why the mistaken example is valuable even if its score is high. Examples were selected for explanation after frozen evaluation.');
for (const [i, row] of examples.entries()) {
  const left = 68 + i * 589;
  await image(samples, path.join(bundle, row.sample_file), left, 158, 552, 296, `Saved held-out ${row.purpose} example`);
  text(samples, row.purpose === 'success' ? 'Correct prediction' : 'Incorrect prediction', left, 475, 552, 48, 22, { bold: true, color: green });
  text(samples, `Reference  ${pretty(row.actual_class)}\nPrediction  ${pretty(row.predicted_class)}\nModel score  ${percentage(row.score)}`, left, 526, 552, 119, 17);
}

const limits = slide('Limits and next improvement', 'Explain source provenance and why small class counts matter. Priority: collect verified field examples for sparse categories and record plant/session IDs. Identify actual team responsibilities and disclose Codex assistance. Use docs/contributions.md for the record.');
lines(limits, [['Rare classes', 'Sparse examples make individual class results uncertain.'], ['Field coverage', 'The 183-photo supplement covers three tomato labels and shares a source with the master.'],
  ['Next improvement', 'Collect independently labeled field photos for sparse classes and record plant and session IDs.'],
  ['Reproduction', 'Run the archive locally. The review notebook explains model selection.']], { step: 108, textSize: 21 });
text(limits, 'github.com/jacob-amey/ML-Final-Project', 66, 622, 1150, 45, 17, { color: green });

for (let i = 0; i < deck.slides.items.length; i++) {
  const bitmap = await deck.export({ slide: deck.slides.items[i], format: 'png', scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${i+1}.png`), new Uint8Array(await bitmap.arrayBuffer()));
}
const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({ workspaceDir: root, candidatePath, finalPath, pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(toolsDir, 'inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(toolsDir, 'inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--require-native-table-slide', '5'],
  explicitTotalSlideCount: 8, requiredNativeTableOwnerSlides: [5], fontPolicy: { basis: 'design', families: ['Arial'] },
  verifyArtifactToolImport: true, receiptPath: path.join(buildDir, 'validation.json') });
