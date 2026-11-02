"""Build a dependency-light static documentation website from repository docs."""
from pathlib import Path
import html
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'website/dist'
OUT.mkdir(parents=True,exist_ok=True)
from markdown_it import MarkdownIt
md=MarkdownIt('commonmark',{'html':False})
PAGES=[('quickstart','Quickstart'),('installation','Install on your hardware'),('commands','CLI commands'),('behaviours','Counting & dwell'),('python-api','Python API'),('model-contracts','Model preparation'),('benchmarks','Benchmarks'),('compatibility','Compatibility & tests'),('video-lab','Video lab protocol'),('troubleshooting','Troubleshooting'),('commercial-use','Commercial use')]

from site_content import shell as page_shell,home_body,guide_body,research_body,sources_body,LANGUAGES,LOCALES,home_path,start_path

def shell(title,body,active=None):
 return page_shell(title,body,active,kind='lab' if '<div class="video-lab">' in body else 'reference',pages=PAGES)

for slug,title in PAGES:
 source=(ROOT/'docs'/f'{slug}.md').read_text(encoding='utf-8')
 body=md.render(source)
 body=re.sub(r'href="([\w-]+)\.md([#][^"]*)?"',lambda m:'href="'+m[1]+'.html'+(m[2] or '')+'"',body)
 body=body.replace('href="../benchmarks/','href="/downloads/').replace('href="../examples/video-lab/results/report.json"','href="/lab/report.json"').replace('href="../examples/video-lab/sources.json"','href="/lab/sources.json"').replace('href="../examples/video-lab/annotations.json"','href="/lab/annotations.json"')
 if slug=='benchmarks':
  for filename,measurement_title in [('cpu-v03-benchmark.json','Intel CPU execution'),('cuda-v03-benchmark.json','NVIDIA CUDA execution')]:
   report_path=ROOT/'benchmarks'/filename
   if not report_path.exists():continue
   report=json.loads(report_path.read_text(encoding='utf-8'))
   rows=[]
   for item in report.get('models',[]):
    if item.get('status')=='ok':
     timing=item['timings_ms'];total=timing['total_ms'];inference=timing['inference_transfer_ms']
     observed=item.get('profile',{}).get('requested_provider_observed',False)
     execution=html.escape(item.get('selected_provider','unknown'))+(' · profile verified' if observed else ' · unverified')
     rows.append(f'<tr><td>{html.escape(item["name"])}</td><td>{total["p50"]:.2f} ms</td><td>{total["p95"]:.2f} ms</td><td>{inference["p50"]:.2f} ms</td><td>{execution}</td></tr>')
    else:rows.append(f'<tr><td>{html.escape(item["name"])}</td><td colspan="4">{html.escape(item.get("error","Not measured"))}</td></tr>')
   env=report['environment']
   hardware=html.escape((env.get('nvidia_smi') if report['requested_provider']=='cuda' else env.get('cpu_name')) or 'Hardware recorded in report')
   measured=f'<section class="measured"><p class="eyebrow">MEASURED WITH v0.3 · SHARED HOST</p><h2>{measurement_title}</h2><p>{hardware}. Batch 1, {report["iterations"]} timed iterations, {report["warmup"]} warm-ups, {report["threads_requested"]} ONNX CPU threads. Separate verification profile.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Total p50</th><th>Total p95</th><th>Inference + transfer p50</th><th>Execution</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class="note">Uncontrolled background workload. Synthetic input; latency only. Excludes decode, tracking and rules. No accuracy or idle-device ranking claim.</p><a href="/downloads/{filename}">Download raw report →</a></section>'
   body+=measured
  engine_path=ROOT/'benchmarks/engine-v03-benchmark.json'
  if engine_path.exists():
   engine=json.loads(engine_path.read_text(encoding='utf-8'))
   body+=f'<section class="measured"><h2>Measured rule-engine overhead</h2><p>{engine["workload"]["tracks"]} tracks · 20 rules · {engine["workload"]["batches"]:,} batches. p50 <strong>{engine["evaluation_ms"]["p50"]:.3f} ms</strong>, p95 <strong>{engine["evaluation_ms"]["p95"]:.3f} ms</strong>. Synthetic replay; excludes perception and serialization.</p><a href="/downloads/engine-v03-benchmark.json">Download raw engine report →</a></section>'
 headings=re.findall(r'<h2>(.*?)</h2>',body)
 (OUT/'docs').mkdir(exist_ok=True)
 (OUT/'docs'/f'{slug}.html').write_text(shell(title,f'<article><p class="eyebrow">DOCUMENTATION / {html.escape(title.upper())}</p>{body}<div class="article-bottom">Source: <code>docs/{slug}.md</code><a href="/docs/commands.html">Command reference →</a></div></article>',slug),encoding='utf-8')

for code in LANGUAGES:
 locale=LOCALES[code]
 home=OUT/('index.html' if code=='en' else f'{code}/index.html')
 home.parent.mkdir(parents=True,exist_ok=True)
 home.write_text(page_shell(locale['home']['title_a']+' '+locale['home']['title_b'],home_body(code),code=code,kind='home'),encoding='utf-8')
 guide=OUT/('start.html' if code=='en' else f'{code}/start.html')
 guide.write_text(page_shell(locale['guide']['title'],guide_body(code),code=code,kind='start'),encoding='utf-8')
(OUT/'research.html').write_text(page_shell('Developer gaps and product opportunities',research_body(),kind='research'),encoding='utf-8')
(OUT/'sources.html').write_text(page_shell('Source and upstream repositories',sources_body(),kind='sources'),encoding='utf-8')
(OUT/'assets').mkdir(exist_ok=True)
for name in ['style.css','app.js','professional.css','professional.js','video-lab.css','video-lab.js']:
 shutil.copy2(ROOT/'website'/name,OUT/'assets'/name)
shutil.copy2(ROOT/'website/research.json',OUT/'assets/research.json')
# Extract the exact video frame whose actual model evidence is displayed on the home page.
import av,cv2
with av.open(str(ROOT/'examples/video-lab/media/shelf.mp4')) as container:
 for frame in container.decode(video=0):
  if float(frame.time)+1e-8>=3.2:
   if not cv2.imwrite(str(OUT/'assets/home-frame.jpg'),frame.to_ndarray(format='bgr24'),[cv2.IMWRITE_JPEG_QUALITY,90]):raise OSError('home frame could not be written')
   break
 else:raise RuntimeError('home evidence frame was not available')
(OUT/'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#2045dd"/><path d="M10 7v18h5a6 6 0 1 0 0-12h-1V7h-4zm4 10h1a2 2 0 1 1 0 4h-1v-4z" fill="white"/></svg>',encoding='utf-8')
(OUT/'downloads').mkdir(exist_ok=True)
for path in (ROOT/'benchmarks').glob('*.json'):
 shutil.copy2(path,OUT/'downloads'/path.name)
for path in (ROOT/'dist').glob('*.whl'):
 shutil.copy2(path,OUT/'downloads'/path.name)
for path in (ROOT/'benchmarks/validation').glob('*'):
 (OUT/'downloads/validation').mkdir(exist_ok=True)
 shutil.copy2(path,OUT/'downloads/validation'/path.name)
archive=ROOT.parent/'behaviorkit-prototype-source.zip'
if archive.exists():shutil.copy2(archive,OUT/'downloads'/archive.name)
lab=ROOT/'examples/video-lab/results'
if (lab/'report.json').exists():
 (OUT/'lab/media').mkdir(parents=True,exist_ok=True)
 shutil.copy2(lab/'report.json',OUT/'lab/report.json')
 shutil.copy2(ROOT/'examples/video-lab/media/manifest.json',OUT/'lab/sources.json')
 shutil.copy2(ROOT/'examples/video-lab/annotations.json',OUT/'lab/annotations.json')
 report=json.loads((lab/'report.json').read_text(encoding='utf-8'))
 if report.get('status')!='completed_execution_not_accuracy_certification':raise RuntimeError('Video lab export is incomplete or failed; refusing to publish partial evidence')
 for case in report['cases']:
  for source in [ROOT/'examples/video-lab/media'/f'{case["id"]}.mp4',lab/'media'/f'{case["id"]}.jpg']:
   if not source.exists():raise FileNotFoundError('Prepare demo videos with scripts/prepare_video_demo.py and build_video_demo.py first: '+str(source))
   shutil.copy2(source,OUT/'lab/media'/source.name)
  for model in report['models']:
   source=lab/'data'/case['id']/model;dest=OUT/'lab/data'/case['id']/model;dest.mkdir(parents=True,exist_ok=True)
   for filename in ['frames.json','summary.json','observations.jsonl','config.json']:shutil.copy2(source/filename,dest/filename)
   for filename in ['events.jsonl','counts.jsonl']:shutil.copy2(source/'replay'/filename,dest/filename)
 (OUT/'demo.html').write_text(shell('Video lab: inspect the evidence',(ROOT/'website/video-lab.html').read_text(encoding='utf-8')),encoding='utf-8')
print(json.dumps({'pages':len(list(OUT.rglob('*.html'))),'languages':len(LANGUAGES),'directory':str(OUT),'video_lab_included':(OUT/'demo.html').exists()}))
