// 两版HTML共用，导出时内联，不访问外部服务。
function structureHTML(detail) {
 if(!detail)return '';
 const names={fact:'独立事实',real_event:'现实事件',learning_episode:'学习经历',shared_work:'共同作品',request:'当次需求',visual:'画面观察',real:'现实',story:'故事',game:'游戏',photo:'照片',attempt:'作答',hint:'提示 / 题目',feedback:'反馈（非外部核实）',contribution:'用户贡献',event:'事件',context:'助手叙事语境（不计事实支持）'};
 return '<details class="structure"><summary>'+esc(names[detail.type]||detail.type)+' · '+esc(detail.topic||'未标注主题')+'</summary><p>场景：'+esc(names[detail.context]||detail.context)+'</p><ol>'+detail.steps.map(s=>'<li><b>'+esc(s.speaker==='user'?'用户':'助手语境')+' · '+esc(names[s.kind]||s.kind)+'</b><blockquote>'+esc(s.text)+'</blockquote><small style="overflow-wrap:anywhere">'+esc(s.evidence_id)+'</small></li>').join('')+'</ol></details>';
}
function summariesHTML(rows) {
 return '<h3>跨会话观察总结</h3><p>只汇总有效记录，不替代原始记忆，也不等于长期喜好。</p>'+((rows||[]).map(s=>'<article class="line box" style="overflow-wrap:anywhere"><b>'+esc(s.claim)+'</b><p>'+esc(s.boundary)+'</p><small>观察范围：'+esc(s.window_start)+' — '+esc(s.window_end)+' · 版本 '+s.revision+'</small><details><summary>来源关联 · '+s.source_refs.length+' 条记忆</summary>'+s.source_refs.map(r=>'<p>'+esc(r.memory_id)+' / v'+r.revision+'<br>'+r.evidence_ids.map(esc).join('<br>')+'</p>').join('')+'</details></article>').join('')||'<p>尚无达到独立会话阈值的已审核主题记录。</p>');
}
function supportHTML(items){return (items||[]).filter(x=>x.support_kind==='base_only').map(x=>'<p class="meta">新增证据仅支持基础喜好，不强化“最喜欢”等程度：'+esc(x.claim)+'</p>').join('')}
function taskHTML(tasks){return (tasks||[]).length?'<section class="notice"><h3>技术任务待恢复（不等于来源不可信）</h3>'+tasks.map(t=>'<p>'+esc(t.session_id||'')+' · '+esc(t.stage)+'：'+esc(t.reason)+'</p>').join('')+'</section>':''}
function executionHTML(data){
 if(data.flow_version!=='generic-flow-v2')return '';
 const s=data.execution_summary||{}, labels={new:'新增',reinforce:'强化',coexist:'并存',supersede:'更新',conflict:'冲突',semantic_pending:'语义待确认',awaiting_commit:'尚未提交',rejected:'拒绝',discarded:'不保留',storage_blocked:'禁止保存'};
 const types={memory:'长期记忆',event:'事件',session_record:'当次记录',unknown:'旧结果未标注'};
 const versions='<p>处理版本（Session数）：'+Object.entries(s.processing_versions||{}).map(([k,v])=>esc(k)+' '+esc(v)).join(' · ')+(s.mixed_versions?'。包含旧已提交结果与新恢复结果，旧结果未经重审。':'')+'</p>';
 const retention='<p>已入库候选去向（含强化，非去重后的记忆条数）：'+Object.entries(s.admitted_candidate_retention||{}).map(([k,v])=>esc(types[k]||k)+' '+esc(v)).join(' · ')+'</p>';
 const invalid='<details><summary>缓存失效原因</summary>'+(s.cache_invalidations||[]).map(v=>'<p>'+esc(v.session_id)+' · '+esc(v.stage)+'：'+esc(v.reason)+'</p>').join('')+'</details>';
 return '<section class="notice"><h3>执行完整性</h3><p>已遍历 '+esc(s.traversed_sessions||0)+' · 已提交 '+esc(s.committed_sessions||0)+' · 技术待处理 '+esc(s.technical_pending_sessions||0)+' · 未遍历 '+esc(s.unvisited_sessions||0)+'</p><p>'+Object.entries(s.candidate_outcomes||{}).map(([k,v])=>esc(labels[k]||k)+' '+esc(v)).join(' · ')+'</p>'+retention+versions+invalid+'<details><summary>实际模型调用统计（包含失败重试）</summary><p>调用为累计日志；缓存为当前Session结果快照。</p>'+Object.entries(data.actual_stage_metrics||{}).map(([k,v])=>'<p>'+esc(k)+'：'+esc(v.calls)+' 次，其中重试 '+esc(v.retries||0)+' 次，缓存命中 '+esc(v.cache_hits||0)+' 次，缓存失效 '+esc(v.cache_invalidated||0)+' 次，已知 token '+esc(v.known_tokens)+'，用量缺失 '+esc(v.usage_missing)+' 次，'+esc(Number(v.elapsed_seconds||0).toFixed(2))+' 秒</p>').join('')+'</details></section>';
}
