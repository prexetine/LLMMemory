// 本地离线产物的浏览器冒烟测试；无需网络或安装依赖。
const {chromium}=require(process.env.MEMORY_PLAYWRIGHT_PATH||'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {pathToFileURL}=require('url');
const path=require('path');
(async()=>{
 const folder=process.argv[2];if(!folder)throw Error('缺少待验证目录');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const errors=[];
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(folder,'index.html')).href);
  for(const view of ['event','session_record','memories']){
   await page.locator('[data-view="'+view+'"]').click();
   if(!await page.locator('#content').isVisible())throw Error('分类视图不可见：'+view);
  }
  await page.locator('[data-view="maintenance"]').click();
  if(!await page.locator('#content').isVisible())throw Error('版本维护视图不可见');
  const options=await page.locator('#user-select option').count();
  if(await page.locator('[data-view="summaries"]').count()){
   await page.locator('[data-view="summaries"]').click();
   if(!await page.locator('#content').getByText('跨会话观察总结',{exact:true}).count())throw Error('总结入口缺失');
  }
  const episode=await page.evaluate(()=>{const p=JSON.parse(document.getElementById('memory-data').textContent);return (p.users||[p])[0].memories.find(m=>m.structure?.type==='learning_episode')?.id});
  if(episode){
   await page.locator('[data-view="event"]').click();
   await page.locator('[data-memory="'+episode+'"]').click();
   if(!await page.locator('#body .structure').count())throw Error('正式版学习明细缺失');
   await page.locator('#body .structure summary').click();
   const hasFeedback=await page.evaluate(id=>{const p=JSON.parse(document.getElementById('memory-data').textContent);return (p.users||[p])[0].memories.find(m=>m.id===id)?.structure.steps.some(s=>s.kind==='feedback')},episode);
   if(hasFeedback&&!await page.locator('#body .structure').innerText().then(t=>t.includes('非外部核实')))throw Error('助手反馈边界缺失');
   await page.locator('#close').click();
  }
  if(options>1)await page.locator('#user-select').selectOption('1');
  await page.screenshot({path:path.join(folder,'formal-preview.png'),fullPage:true});
  await page.goto(pathToFileURL(path.join(folder,'index_report.html')).href);
  await page.locator('[data-tab="memories"]').first().click();
  for(const status of ['pending','rejected','conflicted','superseded','event','session_record','discarded','merged']){
   await page.locator('#statusFilter').selectOption(status);
   await page.locator('#statusFilter').dispatchEvent('input');
  }
  await page.locator('#statusFilter').selectOption('');
  await page.locator('#statusFilter').dispatchEvent('input');
  if(episode&&!await page.locator('.card .structure').count())throw Error('汇报版学习明细缺失');
  if(await page.locator('#userSelect option').count()>1)await page.locator('#userSelect').selectOption('1');
  await page.screenshot({path:path.join(folder,'report-preview.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2);
  if(overflow)throw Error('移动端整体横向溢出');
  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({html:'passed',users:options,states:9,mobile:'passed',errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
