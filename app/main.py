from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import asyncio

from app.routes import upload
from app.services.knowledge import ingest_knowledge_base, knowledge_collection

from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print(" Starting FastAPI server...")

    try:
        print(" Ingesting knowledge base...")
        new_chunks = await asyncio.to_thread(ingest_knowledge_base)
        
        total_chunks = knowledge_collection.count()
        
        if new_chunks > 0:
            print(f" Knowledge base updated — {new_chunks} new chunks ingested")
        else:
            print("Knowledge base is up to date (no new chunks)")
            
        print(f"Total knowledge chunks available: {total_chunks}")
        
    except Exception as e:
        print(f"[KNOWLEDGE STARTUP ERROR] {e}")

    yield

    #  SHUTDOWN 
    print("Server shutting down...")


app = FastAPI(lifespan=lifespan, title="Your AI Coder")

# Include routers
app.include_router(upload.router)


# Optional: Health check endpoint
@app.get("/health")
async def health():
    try:
        total = knowledge_collection.count()
        return {
            "status": "healthy",
            "knowledge_chunks": total
        }
    except:
        return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "Server is running"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """<!DOCTYPE html>
<html>
<head>
<title>AI Code Agent</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column;overflow:hidden}
  #header{padding:10px 16px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:12px;background:#161b22;flex-shrink:0}
  #header h1{font-size:14px;font-weight:600;color:#f0f6fc}
  #upload-bar{display:flex;align-items:center;gap:8px;margin-left:auto;flex-wrap:wrap}
  #upload-bar input[type=file]{font-size:11px;color:#8b949e;max-width:150px}
  .btn-g{padding:4px 12px;font-size:12px;background:#238636;color:#fff;border:none;border-radius:5px;cursor:pointer}
  .btn-g:hover{background:#2ea043}
  .btn-g:disabled{background:#21262d;color:#8b949e;cursor:not-allowed}
  .btn-b{padding:4px 12px;font-size:12px;background:#1f6feb;color:#fff;border:none;border-radius:5px;cursor:pointer}
  .btn-b:hover{background:#388bfd}
  .btn-b:disabled{background:#21262d;color:#8b949e;cursor:not-allowed}
  #project-badge{font-size:11px;color:#8b949e;background:#21262d;padding:3px 8px;border-radius:10px;display:none}
  #main{display:flex;flex:1;overflow:hidden}
  #left{width:230px;border-right:1px solid #21262d;display:flex;flex-direction:column;background:#161b22;flex-shrink:0;overflow:hidden}
  #prompt-area{padding:10px;border-bottom:1px solid #21262d;flex-shrink:0}
  #prompt-area label{font-size:10px;color:#8b949e;display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
  #question{width:100%;height:90px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;font-size:12px;padding:6px;border-radius:5px;resize:none;font-family:inherit}
  #question:focus{outline:none;border-color:#388bfd}
  #run-btn{margin-top:6px;width:100%;padding:7px;background:#1f6feb;color:#fff;border:none;border-radius:5px;font-size:12px;cursor:pointer;font-weight:500}
  #run-btn:hover{background:#388bfd}
  #run-btn:disabled{background:#21262d;color:#8b949e;cursor:not-allowed}
  #pipeline{padding:10px;border-bottom:1px solid #21262d;flex-shrink:0}
  #pipeline label{font-size:10px;color:#8b949e;display:block;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}
  .step{display:flex;align-items:center;gap:6px;padding:3px 5px;border-radius:4px;margin-bottom:2px;font-size:11px}
  .step.idle{color:#484f58}
  .step.running{color:#e3b341;background:#272115}
  .step.done{color:#3fb950;background:#0f2d1a}
  .step.failed{color:#f85149;background:#2d0f0f}
  .step.skipped{color:#484f58;background:#1c2128}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
  #file-tree{flex:1;overflow-y:auto;padding:10px;min-height:0}
  #file-tree label{font-size:10px;color:#8b949e;display:block;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}
  .file-item{font-size:11px;color:#8b949e;padding:2px 4px;cursor:pointer;border-radius:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .file-item:hover{color:#c9d1d9;background:#21262d}
  #right{flex:1;display:flex;flex-direction:column;overflow:hidden}
  #tabs{display:flex;border-bottom:1px solid #21262d;background:#161b22;padding:0 12px;flex-shrink:0;align-items:center;gap:2px}
  .tab{padding:8px 12px;font-size:11px;cursor:pointer;color:#8b949e;border-bottom:2px solid transparent;white-space:nowrap}
  .tab.active{color:#f0f6fc;border-bottom-color:#f78166}
  #tab-meta{margin-left:auto;display:flex;gap:6px;align-items:center;padding:2px 0}
  .badge{font-size:10px;padding:2px 7px;border-radius:10px;display:none}
  #panels{flex:1;overflow:hidden}
  .panel{display:none;height:100%;overflow-y:auto;padding:14px}
  .panel.active{display:block}
  .panel pre{background:#161b22;border:1px solid #21262d;border-radius:5px;padding:12px;font-size:12px;font-family:'SFMono-Regular',Consolas,monospace;white-space:pre-wrap;word-break:break-word;line-height:1.6;color:#c9d1d9}
  .empty-state{color:#484f58;font-size:12px;padding:30px 0;text-align:center}
  #status-bar{padding:5px 14px;font-size:10px;color:#8b949e;background:#161b22;border-top:1px solid #21262d;flex-shrink:0}
  #save-bar{padding:8px 14px;background:#161b22;border-top:1px solid #21262d;display:none;flex-shrink:0;gap:6px;align-items:center}
  #save-path{flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;font-size:12px;padding:5px 8px;border-radius:4px;font-family:monospace}
  .msg-row{padding:5px 8px;border-bottom:1px solid #21262d;font-size:11px;display:flex;gap:8px}
  .msg-agent{color:#388bfd;font-weight:500;min-width:80px;flex-shrink:0}
  .msg-type{color:#6e7681;min-width:60px;flex-shrink:0}
  .msg-content{color:#c9d1d9;flex:1;word-break:break-word}
  .tool-item{font-size:11px;color:#8b949e;background:#161b22;border:1px solid #21262d;border-radius:4px;padding:4px 8px;margin-bottom:3px;font-family:monospace}
</style>
</head>
<body>
<div id="header">
  <h1>AI Code Agent</h1>
  <div id="upload-bar">
    <input type="file" id="zip-file" accept=".zip"/>
    <button class="btn-g" id="upload-btn" onclick="uploadZip()">Upload Project</button>
    <span id="project-badge"></span>
  </div>
</div>
<div id="main">
  <div id="left">
    <div id="prompt-area">
      <label>Prompt</label>
      <textarea id="question" placeholder="write a login function...&#10;fix the bug in auth.py&#10;explain how caching works"></textarea>
      <button id="run-btn" onclick="runQuery()">Run</button>
    </div>
    <div id="pipeline">
      <label>Pipeline</label>
      <div class="step idle" id="step-intent"><span class="dot"></span>Intent</div>
      <div class="step idle" id="step-supervisor"><span class="dot"></span>Supervisor</div>
      <div class="step idle" id="step-planner"><span class="dot"></span>Planner</div>
      <div class="step idle" id="step-retriever"><span class="dot"></span>Retriever</div>
      <div class="step idle" id="step-rag_grader"><span class="dot"></span>RAG Grader</div>
      <div class="step idle" id="step-corrective"><span class="dot"></span>Corrective RAG</div>
      <div class="step idle" id="step-coder"><span class="dot"></span>Coder</div>
      <div class="step idle" id="step-verifier"><span class="dot"></span>Verifier</div>
      <div class="step idle" id="step-reviewer"><span class="dot"></span>Reviewer</div>
      <div class="step idle" id="step-learn"><span class="dot"></span>Learn</div>
    </div>
    <div id="file-tree">
      <label>Files</label>
      <div id="file-list"><div class="empty-state" style="padding:6px 0;font-size:11px">No project</div></div>
    </div>
  </div>
  <div id="right">
    <div id="tabs">
      <div class="tab active" id="tab-answer" onclick="showTab('answer')">Code</div>
      <div class="tab" id="tab-plan" onclick="showTab('plan')">Plan</div>
      <div class="tab" id="tab-review" onclick="showTab('review')">Review</div>
      <div class="tab" id="tab-tools" onclick="showTab('tools')">Tools</div>
      <div class="tab" id="tab-agents" onclick="showTab('agents')">Agent Log</div>
      <div id="tab-meta">
        <span class="badge" id="intent-badge"></span>
        <span class="badge" id="conf-badge"></span>
        <button class="btn-b" id="save-btn" onclick="toggleSave()" style="display:none;font-size:11px;padding:3px 10px">Save</button>
      </div>
    </div>
    <div id="panels">
      <div class="panel active" id="panel-answer"><div class="empty-state">Upload a project and write a prompt</div></div>
      <div class="panel" id="panel-plan"><div class="empty-state">Plan will appear here</div></div>
      <div class="panel" id="panel-review"><div class="empty-state">Review will appear here</div></div>
      <div class="panel" id="panel-tools"><div class="empty-state">Tool calls will appear here</div></div>
      <div class="panel" id="panel-agents"><div class="empty-state">Agent communication log will appear here</div></div>
    </div>
    <div id="save-bar">
      <input id="save-path" placeholder="auth/login.py"/>
      <button class="btn-g" onclick="saveFile()">Save</button>
      <button class="btn-b" onclick="toggleSave()">Cancel</button>
    </div>
  </div>
</div>
<div id="status-bar">Ready</div>
<script>
let projectId='',lastAnswer='',stepTimers=[];
const STEPS=['intent','supervisor','planner','retriever','rag_grader','corrective','coder','verifier','reviewer','learn'];
const DELAYS=[0,1000,2500,4000,6000,7500,9500,11500,13000,15000];

function showTab(n){
  ['answer','plan','review','tools','agents'].forEach(x=>{
    document.getElementById('tab-'+x).classList.toggle('active',x===n);
    document.getElementById('panel-'+x).classList.toggle('active',x===n);
  });
}
function setStep(id,st){const e=document.getElementById('step-'+id);if(e)e.className='step '+st;}
function resetSteps(){STEPS.forEach(s=>setStep(s,'idle'));}
function setStatus(m){const e=document.getElementById('status-bar');if(e)e.textContent=m;}
function clearTimers(){stepTimers.forEach(clearTimeout);stepTimers=[];}
function setPanel(id,content){
  const e=document.getElementById(id);if(!e)return;
  if(!content||!content.trim()){e.innerHTML='<div class="empty-state">(empty)</div>';return;}
  e.innerHTML='<pre>'+escHtml(content)+'</pre>';
}
function toggleSave(){
  const b=document.getElementById('save-bar');
  b.style.display=b.style.display==='flex'?'none':'flex';
}
async function saveFile(){
  const path=document.getElementById('save-path').value.trim();
  if(!path||!lastAnswer){alert('Enter a path and make sure there is output');return;}
  try{
    const r=await fetch('/write-file',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({project_id:projectId,relative_path:path,content:lastAnswer})});
    const d=await r.json();
    if(d.success){setStatus('Saved → '+d.path);document.getElementById('save-bar').style.display='none';refreshFiles();}
    else setStatus('Save failed: '+(d.error||'?'));
  }catch(e){setStatus('Save error: '+e);}
}
async function refreshFiles(){
  if(!projectId)return;
  try{const r=await fetch('/project-files?project_id='+projectId);const d=await r.json();renderFiles(d.files||[]);}catch(e){}
}
function renderFiles(files){
  const e=document.getElementById('file-list');if(!e)return;
  if(!files.length){e.innerHTML='<div class="empty-state" style="font-size:11px">No files</div>';return;}
  e.innerHTML=files.map(f=>`<div class="file-item" onclick="loadFile('${escAttr(f)}')" title="${escAttr(f)}">${escHtml(f)}</div>`).join('');
}
async function loadFile(path){
  if(!projectId)return;
  try{
    const r=await fetch('/read-file?project_id='+projectId+'&path='+encodeURIComponent(path));
    const d=await r.json();
    setPanel('panel-answer',d.content||'');
    showTab('answer');setStatus('Viewing: '+path);
    document.getElementById('save-path').value=path;
  }catch(e){setStatus('Load error: '+e);}
}
async function uploadZip(){
  const fi=document.getElementById('zip-file');
  if(!fi||!fi.files[0]){alert('Select a ZIP first');return;}
  const btn=document.getElementById('upload-btn');
  btn.disabled=true;btn.textContent='Uploading...';setStatus('Uploading...');
  const fd=new FormData();fd.append('file',fi.files[0]);
  try{
    const r=await fetch('/upload-zip',{method:'POST',body:fd});
    const d=await r.json();
    if(d.project_id){
      projectId=d.project_id;
      const b=document.getElementById('project-badge');
      if(b){b.textContent=projectId.slice(0,8)+'...';b.style.display='inline';}
      setStatus('Loaded — '+d.chunks+' chunks');
      renderFiles(d.files||[]);
    }else setStatus('Upload failed');
  }catch(e){setStatus('Upload error: '+e);}
  btn.disabled=false;btn.textContent='Upload Project';
}
async function runQuery(){
  const qe=document.getElementById('question');
  const q=qe?qe.value.trim():'';
  if(!q){alert('Write a prompt');return;}
  if(!projectId){alert('Upload a project first');return;}
  const btn=document.getElementById('run-btn');
  btn.disabled=true;btn.textContent='Running...';
  resetSteps();clearTimers();lastAnswer='';
  ['panel-answer','panel-plan','panel-review','panel-tools','panel-agents'].forEach(p=>setPanel(p,''));
  showTab('answer');
  const sb=document.getElementById('save-btn');if(sb)sb.style.display='none';
  const ib=document.getElementById('intent-badge');if(ib)ib.style.display='none';
  const cb=document.getElementById('conf-badge');if(cb)cb.style.display='none';
  STEPS.forEach((s,i)=>{
    stepTimers.push(setTimeout(()=>{
      if(i>0)setStep(STEPS[i-1],'done');
      setStep(s,'running');
      setStatus(s.replace('_',' ')+' running...');
    },DELAYS[i]));
  });
  try{
    const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({project_id:projectId,question:q})});
    clearTimers();
    const d=await r.json();
    STEPS.forEach(s=>setStep(s,'done'));
    if(d.retrieval_ok)setStep('corrective','skipped');
    lastAnswer=d.answer||'';
    setPanel('panel-answer',lastAnswer);
    setPanel('panel-plan',d.plan||'');
    setPanel('panel-review',d.review||'');
    const tools=d.tool_calls||[];
    const tp=document.getElementById('panel-tools');
    if(tp){
      if(tools.length)tp.innerHTML=tools.map(t=>`<div class="tool-item">${escHtml(t)}</div>`).join('');
      else tp.innerHTML='<div class="empty-state">No tools used</div>';
    }
    const log=d.agent_log||[];
    const ap=document.getElementById('panel-agents');
    if(ap){
      if(log.length){
        ap.innerHTML=log.map(m=>
          `<div class="msg-row"><span class="msg-agent">${escHtml(m.agent)}</span><span class="msg-type">${escHtml(m.type)}</span><span class="msg-content">${escHtml(m.content)}</span></div>`
        ).join('');
      }else ap.innerHTML='<div class="empty-state">No agent messages</div>';
    }
    if(ib&&d.intent){
      const cols={write:'#1f6feb',fix:'#da3633',explain:'#388bfd',general:'#6e7681'};
      ib.textContent=d.intent;ib.style.background=cols[d.intent]||'#21262d';ib.style.color='#fff';ib.style.display='inline';
    }
    if(cb&&d.retrieval_confidence!==undefined){
      const ok=d.retrieval_ok;
      cb.textContent='RAG '+(d.retrieval_confidence*100).toFixed(0)+'%';
      cb.style.background=ok?'#0f2d1a':'#2d0f0f';cb.style.color=ok?'#3fb950':'#f85149';cb.style.display='inline';
    }
    if(sb&&lastAnswer)sb.style.display='inline';
    const synNote=d.syntax_ok===false?' ⚠ syntax error fixed':'';
    const iterNote=d.coder_iterations>1?' ('+d.coder_iterations+' coder iters)':'';
    const fb=d.success===false?' (fallback)':'';
    setStatus('Done'+fb+synNote+iterNote+' — '+d.context_chunks+' chunks — intent:'+d.intent);
    if(!lastAnswer){setStep('reviewer','failed');setStatus('No output'+(d.error?': '+d.error:''));}
  }catch(e){
    clearTimers();
    STEPS.forEach(s=>setStep(s,'failed'));
    setPanel('panel-answer','Request failed: '+String(e));
    setStatus('Request failed');
  }
  btn.disabled=false;btn.textContent='Run';
}
function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function escAttr(s){return String(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
</script>
</body>
</html>"""