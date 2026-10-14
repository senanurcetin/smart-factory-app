/* NEXUS Decision Studio — no synthetic demo numbers.
 * Renders only authenticated source artifacts exposed by existing Flask JSON APIs.
 * This interface is independent from case_study.py and rul_case_study.py templates.
 */
(() => {
  'use strict';
  const isRul = location.pathname.endsWith('/rul');
  const api = isRul ? '/api/rul-case-study' : '/api/case-study';
  const $ = id => document.getElementById(id);
  const fmt = (value, decimals = 2) => typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('en-US', { maximumFractionDigits: decimals, minimumFractionDigits: decimals })
    : 'Not available';
  const pct = (value, decimals = 1) => fmt(Number(value) * 100, decimals) + '%';
  const unit = (value, suffix = '') => value === null || value === undefined ? 'Unavailable' : fmt(Number(value), 2) + suffix;
  const text = (id, value) => { $(id).textContent = String(value ?? 'Unavailable'); };
  const node = (tag, className = '', content = '') => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = String(content ?? '');
    return el;
  };
  const load = async () => {
    const response = await fetch(api, { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error('Could not load verified report data (HTTP ' + response.status + ').');
    return response.json();
  };
  const drawMetrics = metrics => {
    const root = $('metric-grid');
    root.replaceChildren(...metrics.map(m => {
      const el = node('article','metric');
      el.append(node('div','metric-label',m.label),node('div','metric-value',m.value),node('div','metric-note',m.note));
      return el;
    }));
  };
  const drawBars = (root, rows, fmtValue, ceiling) => {
    if (!rows.length) {root.textContent='No measured result available.';return;}
    const safeMax = ceiling || Math.max(...rows.map(r => Math.max(0,Number(r.value)||0)),.00001);
    root.replaceChildren(...rows.map((r,i) => {
      const div = node('div','chart-row');
      const label = node('span','chart-label',r.label);
      const track = node('div','chart-track');
      const bar = node('span',i===0?'best':'');
      bar.style.width = Math.max(0,Math.min(100,Number(r.value)/safeMax*100))+'%';
      track.append(bar);
      const value=node('strong','chart-value',fmtValue(r.value));
      div.append(label,track,value);
      return div;
    }));
  };
  const drawProtocols = pairs => {
    const root=$('protocol');
    root.replaceChildren(...pairs.map(([label,value,note]) => {
      const el=node('div','compare-band');
      el.append(node('strong','',label),node('span','',value),node('small','',note));
      return el;
    }));
  };
  const drawTable = (headers, rows) => {
    const table=node('table','data-table');
    const head=document.createElement('thead'),hr=document.createElement('tr');
    for(const h of headers)hr.append(node('th','',h));
    head.append(hr);
    const body=document.createElement('tbody');
    for(const row of rows){
      const tr=document.createElement('tr');
      for(const cell of row)tr.append(node('td','',cell));
      body.append(tr);
    }
    table.append(head,body);
    $('benchmark-table').replaceChildren(table);
  };
  const drawLimitations=items=>{
    $('limitations').replaceChildren(...items.map(item=>node('li','',item)));
  };
  const setChrome=mode=>{
    $('nav-'+mode).classList.add('active');
    $('nav-'+mode).setAttribute('aria-current','page');
  };
  const maintenance = data => {
    const s=data.summary||{}, d=s.dataset||{},m=s.final_model||{},q=s.review_queue||{};
    setChrome('maintenance');
    text('crumb-page','MAINTENANCE');text('hero-eyebrow','Maintenance / decision quality');
    text('hero-title','Turn model scores into maintenance priorities.');
    text('hero-description','A decision-focused operating room for an offline predictive maintenance benchmark. Prioritize analyst review capacity using model validation evidence, not invented plant telemetry.');
    text('aside-label','Captured failures · review queue');text('aside-number',pct(q.failure_capture_rate));
    text('aside-caption','Of holdout failures found within the top '+pct(q.selected_budget_fraction,0)+' highest-ranked machine states.');
    text('source-title',d.name||'AI4I 2020');text('source-context','UCI dataset · '+Number(d.rows||0).toLocaleString('en-US')+' independent snapshots · stratified holdout.');
    drawMetrics([
      {label:'Holdout PR-AUC',value:fmt(m.pr_auc,4),note:'Imbalanced-class average precision'},
      {label:'Holdout ROC-AUC',value:fmt(m.roc_auc,4),note:'Classification ranking quality'},
      {label:'Holdout F1',value:fmt(m.f1,4),note:'Precision / recall trade-off'},
      {label:'Review yield lift',value:unit(q.yield_lift_vs_random,'×'),note:'Versus random review on holdout'}
    ]);
    text('brief-chart-heading','Failure review economics');
    text('brief-chart-desc','Observed effectiveness of the selected top-'+pct(q.selected_budget_fraction,0)+' review budget.');
    drawBars($('brief-chart'),[
      {label:'Holdout failure capture',value:q.failure_capture_rate},
      {label:'Review yield',value:q.review_yield},
      {label:'Random-review yield',value:q.random_review_yield}
    ],v=>pct(v),1);
    text('takeaway','Review the highest-risk machines first.');
    text('scope-note','AI4I consists of synthetic independent machine snapshots. It cannot prove that an ongoing factory feed has been monitored or that a physical machine has been prevented from failing.');
    text('detail-title','Failure-mode discrimination');text('detail-note','ROC-AUC from per-mode classification, read from the committed summary.');
    const modes=s.per_failure_mode_models?.summary||[];
    drawBars($('detail-chart'),modes.slice().sort((a,b)=>Number(b.roc_auc)-Number(a.roc_auc)).map(r=>({label:r.label,value:Number(r.roc_auc)})),v=>fmt(v,4),1);
    drawProtocols([
      ['Dataset',fmt(Number(d.rows),0).replace('.00','')+' rows','AI4I static states; not repeated asset telemetry'],
      ['Holdout',fmt(Number(d.evaluation_split?.holdout_rows),0).replace('.00','')+' rows',d.evaluation_split?.strategy||'Deterministic split'],
      ['Features',String(s.feature_engineering?.total_features||'—'),'Engineered features derived from observed measurements']
    ]);
    drawTable(['Failure type','ROC-AUC','Capture at 10%'],
      modes.map(r=>[r.label,fmt(r.roc_auc,4),pct(r.capture_rate_at_10pct)]));
    drawLimitations(s.limitations||[]);
  };
  const rul = data => {
    const s=data.summary||{}, d=s.dataset||{},m=s.final_model||{},b=s.baseline||{},gain=s.lift_vs_baseline||{};
    setChrome('rul');
    text('crumb-page','ASSET LIFE');text('hero-eyebrow','Reliability / lifetime inference');
    text('hero-title','Understand the remaining life of each engine.');
    text('hero-description','A prognostics decision studio grounded in NASA C-MAPSS FD001 held-out evaluation. Explain error, benchmark improvement and generalization boundaries without pretending this is a live flight fleet.');
    text('aside-label','RMSE improvement vs naive baseline');text('aside-number',fmt(gain.rmse_reduction_pct,1)+'%');
    text('aside-caption','Reduction in root mean squared error on the official held-out test protocol.');
    text('source-title','NASA C-MAPSS · '+(d.subset||'FD001'));
    text('source-context',String(d.test_units||'?')+' held-out test engines; reference RUL provided by the evaluation dataset.');
    drawMetrics([
      {label:'Model RMSE',value:fmt(m.rmse,2),note:'Cycles · lower is better'},
      {label:'Naive baseline RMSE',value:fmt(b.rmse,2),note:'Constant median-life baseline'},
      {label:'Training engines',value:fmt(d.train_units,0).replace('.00',''),note:'Run-to-failure training population'},
      {label:'Informative sensors',value:fmt(s.sensors?.kept_informative,0).replace('.00',''),note:'From '+(s.sensors?.total||'—')+' recorded sensors'}
    ]);
    text('brief-chart-heading','Prediction error, side by side');
    text('brief-chart-desc','Official held-out test RMSE in engine cycles. Lower bar represents better prediction.');
    drawBars($('brief-chart'),[
      {label:'Naive median baseline',value:Number(b.rmse)},
      {label:'Tuned rolling-window model',value:Number(m.rmse)}
    ],v=>fmt(v,2)+' cycles',Math.max(Number(b.rmse),Number(m.rmse),.0001));
    text('takeaway','Quantify prognosis uncertainty before scheduling work.');
    text('scope-note','FD001 is a simulated turbofan degradation benchmark, not observed airline flight data. Its 125-cycle early-life target cap is a modeling choice; predictions need domain validation before deployment.');
    const card=data.model_selection?.feature_engineering_impact||{};
    text('detail-title','Feature engineering impact');text('detail-note','RMSE using matched tuned hyperparameters, with versus without rolling-window features.');
    drawBars($('detail-chart'),[
      {label:'Raw sensor features',value:Number(card.raw_features?.rmse??0)},
      {label:'Rolling-window features',value:Number(card.enhanced_features?.rmse??m.rmse)}
    ],v=>fmt(v,2)+' cycles',Math.max(Number(card.raw_features?.rmse||0),Number(card.enhanced_features?.rmse||0),1));
    drawProtocols([
      ['Test engines',String(d.test_units||'—'),'One final observed cycle per held-out engine'],
      ['Validation',String(data.model_selection?.hyperparameter_tuning?.cv_folds||'—')+' folds','GroupKFold by engine; no unit leakage'],
      ['Target cap', '125 cycles',s.target?.definition||'Standard C-MAPSS capped target']
    ]);
    drawTable(['Protocol','RMSE (cycles)','PHM08 score'],[
      ['Naive median baseline',fmt(b.rmse,4),fmt(b.phm08_score,2)],
      ['Tuned gradient boosting',fmt(m.rmse,4),fmt(m.phm08_score,2)],
      ['Error reduction',fmt(gain.rmse_reduction_pct,2)+'%','Not additive']
    ]);
    drawLimitations(s.limitations||[]);
  };
  const bindTabs=()=>{
    for(const button of document.querySelectorAll('[data-tab]')){
      button.addEventListener('click',()=>{
        for(const candidate of document.querySelectorAll('[data-tab]')){
          const active=candidate===button;
          candidate.setAttribute('aria-selected',active?'true':'false');
          $('panel-'+candidate.dataset.tab).hidden=!active;
        }
      });
    }
  };
  bindTabs();
  load().then(data=>{isRul?rul(data):maintenance(data);$('status').textContent='';})
    .catch(error=>{
      const status=$('status');status.replaceChildren(node('div','error',error.message));
      text('hero-title','Evidence temporarily unavailable');
      text('hero-description','This preview displays only verified analysis artifacts. No fallback or invented values will be shown.');
      document.querySelectorAll('.workspace-section').forEach(p=>p.hidden=true);
    });
})();
