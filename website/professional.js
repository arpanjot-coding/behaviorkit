(()=>{
 'use strict';
 document.getElementById('language-select')?.addEventListener('change',event=>location.assign(event.target.value));
 const menu=document.getElementById('menu'),header=document.querySelector('.professional-header'),nav=document.getElementById('main-navigation');
 function closeMenu(){header?.classList.remove('menu-open');menu?.setAttribute('aria-expanded','false');}
 menu?.addEventListener('click',()=>{if(menu.getAttribute('aria-expanded')==='true')nav.querySelector('a')?.focus();});
 document.addEventListener('keydown',event=>{if(event.key==='Escape'&&menu?.getAttribute('aria-expanded')==='true'){closeMenu();menu.focus();}});
 window.addEventListener('resize',()=>{if(innerWidth>1180)closeMenu();});
 const tabs=[...document.querySelectorAll('[data-command-tab]')];
 function selectTab(button,focus=false){tabs.forEach(tab=>{const selected=tab===button;tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;document.getElementById(tab.getAttribute('aria-controls')).hidden=!selected;});if(focus)button.focus();}
 tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>selectTab(tab));tab.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(index+1)%tabs.length;else if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=tabs.length-1;else return;event.preventDefault();selectTab(tabs[next],true);});});
 const search=document.getElementById('gap-search'),filters=[...document.querySelectorAll('[data-gap-filter]')],gaps=[...document.querySelectorAll('.research-gap')];let priority='all';
 function filterGaps(){const query=(search?.value||'').trim().toLocaleLowerCase();let count=0;gaps.forEach(gap=>{const visible=(priority==='all'||gap.dataset.priority===priority)&&gap.dataset.search.includes(query);gap.hidden=!visible;if(visible)count++;});const status=document.getElementById('gap-count');if(status)status.textContent=count+' of '+gaps.length+' opportunities';const empty=document.getElementById('gap-empty');if(empty)empty.hidden=count!==0;}
 search?.addEventListener('input',filterGaps);filters.forEach(button=>button.addEventListener('click',()=>{priority=button.dataset.gapFilter;filters.forEach(b=>b.setAttribute('aria-pressed',String(b===button)));filterGaps();}));
})();
