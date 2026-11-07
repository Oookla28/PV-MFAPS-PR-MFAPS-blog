from __future__ import annotations
import json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
D=json.loads((ROOT/'data/research-data.json').read_text())

def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
def svg_open(w=960,h=560): return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">','<style>text{font-family:"Songti SC","STSong","SimSun",serif} .axis{stroke:#6d6962;stroke-width:1} .grid{stroke:#e6dfd4;stroke-width:1} .label{fill:#4b4742;font-size:14px} .small{fill:#766f66;font-size:12px} .title{fill:#292621;font-size:22px;font-weight:600} </style>']
def save(name,lines): (FIG/name).write_text('\n'.join(lines+['</svg>']))
def xmap(x,a,b,left,right): return left+(x-a)/(b-a)*(right-left)
def ymap(y,a,b,top,bottom): return bottom-(y-a)/(b-a)*(bottom-top)
def chart_frame(title,ylabel,xlabel,x_ticks,y_min,y_max, w=960,h=560):
    left,top,right,bottom=92,70,w-36,h-72; out=svg_open(w,h); out.append(f'<text class="title" x="{left}" y="35">{esc(title)}</text>')
    for y in [y_min+(y_max-y_min)*i/5 for i in range(6)]:
        yy=ymap(y,y_min,y_max,top,bottom); out.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/><text class="small" x="{left-12}" y="{yy+4:.1f}" text-anchor="end">{y:.2f}</text>')
    for x,lab in x_ticks:
        xx=xmap(x,x_ticks[0][0],x_ticks[-1][0],left,right); out.append(f'<line class="axis" x1="{xx:.1f}" y1="{bottom}" x2="{xx:.1f}" y2="{bottom+6}"/><text class="small" x="{xx:.1f}" y="{bottom+24}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/><line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    out.append(f'<text class="label" x="{(left+right)/2}" y="{h-14}" text-anchor="middle">{esc(xlabel)}</text><text class="label" transform="translate(20 {(top+bottom)/2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    return out,(left,top,right,bottom)

def fig_pareto():
    arms=D['pr_final']['arms']; colors={'pr_pv_mfaps':'#b85c38','bs_fixed_k_matched_B02':'#5b7fa3','effective_number':'#3b8e68'}
    xs=[a['held_tail10'] for a in arms]; ys=[a['held_macro'] for a in arms]
    out,box=chart_frame('HELD macro–tail frontier','HELD Macro-F1','HELD Tail-10%',[(0.26,'0.26'),(0.28,'0.28'),(0.30,'0.30')],0.44,0.50)
    l,t,r,b=box
    for a in arms:
        xx=xmap(a['held_tail10'],0.26,0.30,l,r); yy=ymap(a['held_macro'],0.44,0.50,t,b); c=colors.get(a['name'],'#666')
        out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="9" fill="{c}"/><text class="small" x="{xx+12:.1f}" y="{yy-10:.1f}">{esc(a["name"])}</text>')
    save('pareto-macro-tail.svg',out)

def fig_epsilon():
    rows=[(1.0,3),(0.5,3),(0.2,0.0),(0.1,0.0),(0.05,1),(0.02,2),(0.01,2),(0.005,2),(0.001,4),(0.0001,4),(0,4)]
    # Use log-like hand positions to keep zero readable.
    positions=[0,1,2,3,4,5,6,7,8,9,10]
    out,box=chart_frame('KL radius changes the discrete policy','changed classes','epsilon',[(0,'1'),(3,'0.1'),(6,'0.01'),(10,'0')],0,5)
    l,t,r,b=box
    pts=[]
    for i,(eps,n) in enumerate(rows):
        xx=xmap(i,0,10,l,r); yy=ymap(n,0,5,t,b); pts.append(f'{xx:.1f},{yy:.1f}')
    out.append(f'<polyline fill="none" stroke="#b85c38" stroke-width="4" points="{" ".join(pts)}"/>')
    for i,(eps,n) in enumerate(rows):
        xx=xmap(i,0,10,l,r); yy=ymap(n,0,5,t,b); out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6" fill="#b85c38"/><text class="small" x="{xx:.1f}" y="{yy-12:.1f}" text-anchor="middle">{n}</text>')
    save('epsilon-action-changes.svg',out)

def fig_system():
    out=svg_open(960,420); out += ['<text class="title" x="70" y="42">Legacy versus fused execution graph</text>']
    for y,label,n,c in [(130,'legacy',3,'#a76c4f'),(290,'fused',1,'#3d8967')]:
        out.append(f'<text class="label" x="70" y="{y+8}">{label}</text>')
        for i in range(n):
            x=235+i*170; out.append(f'<rect x="{x}" y="{y-30}" width="120" height="60" rx="10" fill="{c}" opacity=".9"/><text x="{x+60}" y="{y+6}" text-anchor="middle" fill="white" font-size="15">encoder pass</text>')
        if n==3:
            out.append('<path d="M365 130 H395 M535 130 H565" stroke="#776e65" stroke-width="2" marker-end="url(#a)"/>')
    out += ['<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#776e65"/></marker></defs>', '<text class="small" x="70" y="382">real-checkpoint action/counter identity · state max |Δ| = 4.158×10⁻⁶ · tolerance = 10⁻⁵</text>']
    save('legacy-vs-fused.svg',out)

def fig_timeline():
    out=svg_open(1100,520); out.append('<text class="title" x="70" y="42">The research moved by failed hypotheses</text>')
    events=[('PlantSPD','Raw / Log readout gap','#5b7fa3'),('E0–E2','feedback mismatch diagnostics','#6f5d87'),('3090','long-tail pivot and failed cost gate','#a76c4f'),('A100','action space + fused execution','#3d8967'),('PR','KL robust shared policy','#b85c38'),('HELD','24 states / 25,104 reads','#3b8e68')]
    x0=100; step=170; y=230
    out.append(f'<line x1="{x0}" y1="{y}" x2="{x0+step*(len(events)-1)}" y2="{y}" stroke="#b8ad9f" stroke-width="5"/>')
    for i,(title,sub,c) in enumerate(events):
        x=x0+i*step; out.append(f'<circle cx="{x}" cy="{y}" r="18" fill="{c}"/><text x="{x}" y="{y+55}" text-anchor="middle" class="label">{esc(title)}</text><text x="{x}" y="{y+80}" text-anchor="middle" class="small">{esc(sub)}</text>')
    out.append('<text class="small" x="70" y="440">每一次加法都来自一次对先前假设的拆分。失败实验保留在时间线上，但不进入最终均值。</text>')
    save('research-timeline.svg',out)

def fig_gate():
    out=svg_open(960,480); out.append('<text class="title" x="72" y="40">Partition gate: why PR was registered</text>')
    vals=[('Macro gap',D['partition_gate']['macro_gap'],D['partition_gate']['thresholds']['macro'],'#3d8967'),('Tail gap',D['partition_gate']['tail_gap'],D['partition_gate']['thresholds']['tail'],'#a76c4f'),('Action TV',D['partition_gate']['action_tv'],D['partition_gate']['thresholds']['action_tv'],'#8a6f9d')]
    for i,(name,v,th,c) in enumerate(vals):
        y=120+i*105; scale=310/max(v,th); out.append(f'<text class="label" x="72" y="{y}">{name}</text><rect x="220" y="{y-20}" width="310" height="28" rx="7" fill="#eee8df"/><rect x="220" y="{y-20}" width="{min(310,v*scale):.1f}" height="28" rx="7" fill="{c}"/><line x1="{220+th*scale:.1f}" y1="{y-28}" x2="{220+th*scale:.1f}" y2="{y+18}" stroke="#3e3933" stroke-dasharray="5 4"/><text class="small" x="555" y="{y+4}">{v:.4f} · threshold {th:.2f}</text>')
    out.append('<text class="small" x="72" y="445">只有 Macro gap 触发了注册。Tail gap 与 action TV 没有超过各自阈值。</text>'); save('partition-gate.svg',out)

def fig_risk():
    out,box=chart_frame('Two partition envelopes for the changed classes','risk proxy','inner steps k',[(0,'0'),(1,'1'),(2,'2'),(4,'4')],0,470)
    l,t,r,b=box
    # values from action-difference analysis
    data={'class 38 p0':[61.5807,61.7580,61.9370,62.3005],'class 38 p1':[14.1952,14.1088,14.0233,13.8553],'class 68 p0':[461.7041,461.5838,461.7418,462.5],'class 68 p1':[252.2245,252.2970,252.4672,253.0]}
    colors=['#b85c38','#5b7fa3','#3d8967','#8a6f9d']
    for (label,vals),c in zip(data.items(),colors):
        pts=[]
        for i,k in enumerate([0,1,2,4]):
            xx=xmap(k,0,4,l,r); yy=ymap(vals[i],0,470,t,b); pts.append(f'{xx:.1f},{yy:.1f}')
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="3" points="{" ".join(pts)}"/><text class="small" x="{r-170}" y="{t+18+18*list(data).index(label)}" fill="{c}">{label}</text>')
    save('risk-envelopes.svg',out)

for f in [fig_pareto,fig_epsilon,fig_system,fig_timeline,fig_gate,fig_risk]: f()
print('generated',len(list(FIG.glob('*.svg'))),'figures')
