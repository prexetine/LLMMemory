// Offline template checks: no browser, model, or database writes.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const template = fs.readFileSync(path.join(__dirname, '../assets/memory_report.html'), 'utf8');
for (const match of template.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
  if (!match[0].includes('application/json')) new vm.Script(match[1]);
}
let selected = {cards:[
  {dimension:'G16',status:'confirmed',claim:'喜欢恐龙'},
  {dimension:'G17',dimension_ids:['G16','G17'],status:'pending',claim:'待确认'},
  {dimension:'G01',status:'confirmed',claim:'不应混入'}
]};
const context = vm.createContext({
  user:()=>selected,
  esc:s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;')
});
const start = template.indexOf('function dimensionMemories');
const end = template.indexOf("document.addEventListener('toggle'", start);
vm.runInContext(template.slice(start,end), context);
assert.equal(vm.runInContext("dimensionMemories('G16').length",context),2);
const result = vm.runInContext("dimensionDetails({id:'G16',name:'喜欢'})",context);
assert.match(result,/当前有效 1 条，其余 1 条/);
assert.match(result,/data-dimension-details="G16"/);
assert.match(vm.runInContext("dimensionDetails({id:'G99',name:'空'})",context),/暂无记忆/);
assert.match(vm.runInContext("dimensionDetails({id:'G16',name:'<危险>'})",context),/&lt;危险>/);
selected={cards:[]};
assert.equal(vm.runInContext("dimensionMemories('G16').length",context),0);
console.log('PASS: script syntax, multi-dimension mapping, status counts, empty state, escaping, user isolation');
const elements = {dimSearch:{value:''},dimRows:{innerHTML:''},dimCount:{textContent:''}};
const kpi = {textContent:''};
context.$ = id=>elements[id];
context.document = {querySelector:()=>kpi};
vm.runInContext(template.slice(template.indexOf('function dimStats'),template.indexOf('\n',template.indexOf('function dimStats'))),context);
vm.runInContext(template.slice(template.indexOf('function renderDims'),template.indexOf('function bars')),context);
selected={
  run_status:'completed',
  dimensions:[{id:'G25',name:'去过',rules:'规则'},{id:'UNCLASSIFIED',name:'未分类',rules:'未匹配'}],
  cards:[{dimension:'G25',status:'event'},{dimension:'G25',status:'session_record'}]
};
vm.runInContext('renderDims()',context);
assert.match(elements.dimRows.innerHTML, /<td>2<\/td><td>0<\/td>/);
assert.equal(kpi.textContent,1);
assert.equal(elements.dimCount.textContent,'1 / 1 个产品维度；未分类单列');
assert.match(elements.dimRows.innerHTML,/有有效记忆\/事件\/当次记录/);
assert.doesNotMatch(elements.dimRows.innerHTML,/<td>4<\/td>/);
console.log('PASS: rendered dimension rows, events counted once, unclassified excluded from dimension KPI');
selected.cards=[{dimension:'G16',status:'confirmed',claim:'喜欢<恐龙>',subject:'我',
  mapping_deferred:[{dimension_id:'G20',reason:'缺少跨日期支持'}]}];
assert.equal(vm.runInContext("dimensionMemories('G20').length",context),0);
const deferred = vm.runInContext("dimensionDetails({id:'G20',name:'长期兴趣'})",context);
assert.match(deferred,/条件待补 1 条 · 不计入已映射/);
assert.match(deferred,/喜欢&lt;恐龙>/);
console.log('PASS: deferred mappings visible without inflating dimension coverage');
