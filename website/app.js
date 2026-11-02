const menu=document.getElementById('menu');
menu?.addEventListener('click',()=>{const open=document.querySelector('.topbar').classList.toggle('menu-open');menu.setAttribute('aria-expanded',String(open));});
const copyLabels=new WeakMap(),copyTimers=new WeakMap();
async function copy(button,value){
 const original=copyLabels.get(button)||button.textContent;copyLabels.set(button,original);clearTimeout(copyTimers.get(button));
 try{await navigator.clipboard.writeText(value);button.textContent=document.documentElement.dataset.copyDone||'Copied';}
 catch{const code=button.closest('.pro-code, pre')?.querySelector('code');if(code){const range=document.createRange();range.selectNodeContents(code);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);}button.textContent=document.documentElement.dataset.copyError||'Select the code to copy';}
 copyTimers.set(button,setTimeout(()=>button.textContent=original,1500));
}
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',()=>copy(button,button.dataset.copy)));
document.querySelectorAll('[data-copy-target]').forEach(button=>button.addEventListener('click',()=>copy(button,document.getElementById(button.dataset.copyTarget).textContent)));
document.querySelectorAll('.docs-main pre').forEach((pre,index)=>{const button=document.createElement('button');button.textContent='Copy';button.setAttribute('aria-label','Copy code example '+(index+1));button.addEventListener('click',()=>copy(button,pre.querySelector('code').textContent));pre.append(button);});
