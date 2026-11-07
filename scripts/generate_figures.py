from __future__ import annotations
import json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
D=json.loads((ROOT/'data/research-data.json').read_text())

def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
def shell(w=1100,h=650):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">', '<style>text{font-family:"Songti SC","STSong","SimSun",serif}.title{fill:#2b2824;font-size:25px;font-weight:600}.axis{stroke:#70675d;stroke-width:1}.grid{stroke:#e5ddd1;stroke-width:1}.label{fill:#4b4540;font-size:15px}.small{fill:#766d63;font-size:13px}.legend{fill:#4b4540;font-size:14px}</style>']
def save(name, lines): (FIG/name).write_text('\n'.join(lines+['</svg>']))
def xmap(x,a,b,l,r): return l+(x-a)/(b-a)*(r-l)
def ymap(y,a,b,t,bot): return bot-(y-a)/(b-a)*(bot-t)
def frame(title,xlabel,ylabel,x_ticks,ymin,ymax,w=1100,h=650):
    l,t,r,bot=100,85,w-55,h-105; out=shell(w,h)
    out.append(f'<text class="title" x="{l}" y="43">{esc(title)}</text>')
    for i in range(6):
        y=ymin+(ymax-ymin)*i/5; yy=ymap(y,ymin,ymax,t,bot)
        out.append(f'<line class="grid" x1="{l}" y1="{yy:.1f}" x2="{r}" y2="{yy:.1f}"/><text class="small" x="{l-12}" y="{yy+4:.1f}" text-anchor="end">{y:.2f}</text>')
    for x,lab in x_ticks:
        xx=xmap(x,x_ticks[0][0],x_ticks[-1][0],l,r)
        out.append(f'<line class="axis" x1="{xx:.1f}" y1="{bot}" x2="{xx:.1f}" y2="{bot+7}"/><text class="small" x="{xx:.1f}" y="{bot+26}" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<line class="axis" x1="{l}" y1="{t}" x2="{l}" y2="{bot}"/><line class="axis" x1="{l}" y1="{bot}" x2="{r}" y2="{bot}"/>')
    out.append(f'<text class="label" x="{(l+r)/2}" y="{h-24}" text-anchor="middle">{esc(xlabel)}</text><text class="label" transform="translate(23 {(t+bot)/2}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    return out,(l,t,r,bot)

def line(out, xs, ys, color, box, ymin, ymax, width=4):
    l,t,r,bot=box; pts=[]
    for x,y in zip(xs,ys): pts.append(f'{xmap(x,min(xs),max(xs),l,r):.1f},{ymap(y,ymin,ymax,t,bot):.1f}')
    out.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{" ".join(pts)}"/>')
    for x,y in zip(xs,ys):
        xx=xmap(x,min(xs),max(xs),l,r); yy=ymap(y,ymin,ymax,t,bot); out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}"/>')

def fig_bias_stagnation():
    ks=[0,1,2,4,8,16]; rho=0.9998
    bias=[rho**(2*k) for k in ks]
    a=math.log(1.173)/16
    stability=[math.exp(a*k) for k in ks]
    out,box=frame('偏差停滞与稳定性代价','内层步数 k','归一化风险代理',[(0,'0'),(1,'1'),(2,'2'),(4,'4'),(8,'8'),(16,'16')],0.96,1.20)
    l,t,r,bot=box
    x4=xmap(4,0,16,l,r); out.append(f'<rect x="{x4:.1f}" y="{t}" width="{r-x4:.1f}" height="{bot-t}" fill="#b85c38" opacity=".08"/>')
    line(out,ks,bias,'#356da8',box,0.96,1.20)
    line(out,ks,stability,'#b85c38',box,0.96,1.20)
    out += [f'<text class="legend" x="{l+25}" y="{t+25}" fill="#356da8">偏差包络 ρ^(2k)，ρ≈{rho:.4f}</text>', f'<text class="legend" x="{l+25}" y="{t+49}" fill="#b85c38">稳定性/EB 经验代理，k=16 增长 17.3%</text>', f'<text class="small" x="{x4+15:.1f}" y="{t+26}">低收益区</text>', f'<text class="small" x="{l}" y="{bot+55}">k=16 的偏差收益约 0.405%，红线是校准中的经验代理，不是无条件的指数稳定性定理。</text>']
    save('bias-stagnation.svg',out)

def fig_dro_flips():
    w,h=1200,650; out=shell(w,h); out.append('<text class="title" x="70" y="43">DRO 约束下的两个边界动作</text>')
    panels=[('Class 38 · k=2 → 0', [61.5807,61.7580,61.9370,62.3005],[14.1952,14.1088,14.0233,13.8553], 'p0 的风险随 k 上升，鲁棒策略放弃 k=2'), ('Class 68 · k=0 → 1',[461.7041,461.5838,461.7418,462.5],[252.2245,252.2970,252.4672,253.0], 'p0 在 k=1 达到较低风险，鲁棒策略选择 k=1')]
    for j,(title,p0,p1,note) in enumerate(panels):
        x0=70+j*570; l,t,r,bot=x0+65,100,x0+500,520
        out.append(f'<text class="label" x="{x0+65}" y="75">{esc(title)}</text>')
        lo=min(p0+p1)-2; hi=max(p0+p1)+2
        for i in range(6):
            y=lo+(hi-lo)*i/5; yy=ymap(y,lo,hi,t,bot); out.append(f'<line class="grid" x1="{l}" y1="{yy:.1f}" x2="{r}" y2="{yy:.1f}"/><text class="small" x="{l-8}" y="{yy+4:.1f}" text-anchor="end">{y:.1f}</text>')
        for i,k in enumerate([0,1,2,4]):
            xx=xmap(k,0,4,l,r); out.append(f'<line class="axis" x1="{xx:.1f}" y1="{bot}" x2="{xx:.1f}" y2="{bot+6}"/><text class="small" x="{xx:.1f}" y="{bot+23}" text-anchor="middle">{k}</text>')
        out += [f'<line class="axis" x1="{l}" y1="{t}" x2="{l}" y2="{bot}"/><line class="axis" x1="{l}" y1="{bot}" x2="{r}" y2="{bot}"/>']
        for idx,(vals,color,label) in enumerate([(p0,'#b85c38','p0'),(p1,'#356da8','p1')]):
            pts=[]
            for k,v in zip([0,1,2,4],vals): pts.append(f'{xmap(k,0,4,l,r):.1f},{ymap(v,lo,hi,t,bot):.1f}')
            out.append(f'<polyline fill="none" stroke="{color}" stroke-width="4" points="{" ".join(pts)}"/>')
            out.append(f'<text class="legend" x="{r-60}" y="{t+22+22*idx}" fill="{color}">{label}</text>')
        out.append(f'<text class="small" x="{x0+65}" y="{560}">{esc(note)}</text>')
    out.append('<text class="small" x="70" y="620">两个分区的逐类 FIT counts 完全配对，动作翻转由风险曲率决定，不能用频率差异解释。</text>')
    save('dro-action-flips.svg',out)

def fig_risk_heatmap():
    rows=[('38 p0',[61.5807,61.7580,61.9370,62.3005]),('38 p1',[14.1952,14.1088,14.0233,13.8553]),('68 p0',[461.7041,461.5838,461.7418,462.5]),('68 p1',[252.2245,252.2970,252.4672,253.0])]
    w,h=900,460; out=shell(w,h); out.append('<text class="title" x="70" y="43">边界类别的分区风险热力图</text>')
    x0,y0=225,105; cw,ch=120,62
    vals=[v for _,a in rows for v in a]; mn,mx=min(vals),max(vals)
    for j,k in enumerate([0,1,2,4]): out.append(f'<text class="label" x="{x0+j*cw+cw/2}" y="82" text-anchor="middle">k={k}</text>')
    for i,(name,a) in enumerate(rows):
        out.append(f'<text class="label" x="{x0-18}" y="{y0+i*ch+38}" text-anchor="end">{name}</text>')
        for j,v in enumerate(a):
            q=(v-mn)/(mx-mn); color=f'rgb({int(245-90*q)},{int(239-120*q)},{int(225-130*q)})'
            out.append(f'<rect x="{x0+j*cw}" y="{y0+i*ch}" width="{cw-5}" height="{ch-5}" rx="5" fill="{color}"/><text x="{x0+j*cw+cw/2-2}" y="{y0+i*ch+35}" text-anchor="middle" fill="#2b2824" font-size="14">{v:.2f}</text>')
    out.append('<text class="small" x="70" y="410">颜色越深表示风险代理越高。Class 38 的 k=2 与 Class 68 的 k=1 分别由 p0 风险面主导。</text>')
    save('risk-curvature-heatmap.svg',out)

def fig_fused():
    w,h=1200,600; out=shell(w,h); out.append('<text class="title" x="70" y="43">Algorithm–system co-design：逻辑动作与物理执行</text>')
    out += ['<text class="label" x="70" y="110">Legacy</text>','<text class="label" x="70" y="270">Fused</text>']
    for i,x in enumerate([220,430,640]): out.append(f'<rect x="{x}" y="80" width="150" height="60" rx="10" fill="#b85c38"/><text x="{x+75}" y="116" text-anchor="middle" fill="white" font-size="15">Encoder pass {i+1}</text>')
    for x in [370,580]: out.append(f'<path d="M{x} 110 H{x+55}" stroke="#7a7168" stroke-width="3" marker-end="url(#arrow)"/>')
    out.append('<rect x="365" y="240" width="260" height="60" rx="10" fill="#3d8967"/><text x="495" y="276" text-anchor="middle" fill="white" font-size="15">one fused encoder graph</text>')
    out.append('<path d="M520 140 V235" stroke="#7a7168" stroke-width="3" stroke-dasharray="6 5" marker-end="url(#arrow)"/>')
    out += ['<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#7a7168"/></marker></defs>']
    # subplots
    out.append('<text class="label" x="70" y="385">物理 encoder forward</text>')
    for x,val,label,color in [(260,3,'Legacy','#b85c38'),(430,1,'Fused','#3d8967')]:
        height=val*45; y0=520-height; out.append(f'<rect x="{x}" y="{y0}" width="90" height="{height}" fill="{color}"/>')
        out.append(f'<text class="label" x="{x+45}" y="545" text-anchor="middle">{label}</text><text class="label" x="{x+45}" y="{y0-8}" text-anchor="middle">{val}</text>')
    out.append('<text class="label" x="670" y="385">CUDA event time</text>')
    for x,val,label,color in [(850,0.602,'Legacy','#b85c38'),(1000,0.157,'Fused','#3d8967')]:
        height=val/0.602*135; y0=520-height; out.append(f'<rect x="{x}" y="{y0}" width="75" height="{height}" fill="{color}"/><text class="small" x="{x+37}" y="545" text-anchor="middle">{label}</text><text class="small" x="{x+37}" y="{y0-8}" text-anchor="middle">{val:.3f}s</text>')
    out.append('<text class="small" x="70" y="585">真实 checkpoint 等价性：action/counter identity，state max |Δ| = 5.77×10⁻⁶，物理 forward 3→1。</text>')
    save('fused-co-design.svg',out)

def fig_pareto():
    points=[('balanced softmax',0.4727,0.2659,0.555,'#6c7f96'),('fixed-k',0.4553,0.2659,0.630,'#8d8a84'),('bs-pcfa',0.4773,0.2738,0.571,'#3d8967'),('native+PV',0.4653,0.2778,0.560,'#8a6f9d'),('effective number',0.4760,0.2778,0.537,'#356da8'),('PR-PV-MFAPS',0.4841,0.2976,0.606,'#b85c38')]
    out,box=frame('HELD Macro–Tail measured frontier','HELD Macro-F1','HELD Tail-10%',[(0.45,'0.45'),(0.46,'0.46'),(0.47,'0.47'),(0.48,'0.48'),(0.49,'0.49')],0.25,0.31)
    l,t,r,bot=box; mn,mx=0.53,0.64
    for name,x,y,wall,color in points:
        xx=xmap(x,0.45,0.49,l,r); yy=ymap(y,0.25,0.31,t,bot); rad=8+(mx-wall)/(mx-mn)*11
        out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{rad:.1f}" fill="{color}" opacity=".88"/><text class="small" x="{xx+12:.1f}" y="{yy-10:.1f}">{esc(name)}</text>')
    out.append(f'<text class="small" x="{l}" y="{bot+55}">点大小反映 wall-clock，PR-PV-MFAPS 的 fixed-k matched point 位于本次协议的右上边界。</text>')
    save('pareto-frontier.svg',out)

def fig_timeline():
    out=shell(1200,480); out.append('<text class="title" x="70" y="43">从病害检测到反馈几何的研究路径</text>')
    events=[('病害检测','C0+S2 / 重采样','#6c7f96'),('SPD','Raw / Log / 协方差','#74618b'),('机制','读出反馈与差商','#356da8'),('分配','风险包络与 DP','#b85c38'),('系统','Fused 3→1','#3d8967'),('分区','KL-DRO / PR','#9b7447'),('结果','HELD once','#3b8e68')]
    x0,step,y=90,170,205; out.append(f'<line x1="{x0}" y1="{y}" x2="{x0+step*(len(events)-1)}" y2="{y}" stroke="#c2b6a5" stroke-width="5"/>')
    for i,(a,b,c) in enumerate(events):
        x=x0+i*step; out.append(f'<circle cx="{x}" cy="{y}" r="17" fill="{c}"/><text class="label" x="{x}" y="{y+55}" text-anchor="middle">{esc(a)}</text><text class="small" x="{x}" y="{y+79}" text-anchor="middle">{esc(b)}</text>')
    out.append('<text class="small" x="70" y="420">每一轮实验把研究对象从工程效果推向可测机制，再推向可执行的分区鲁棒策略。</text>')
    save('research-timeline.svg',out)

fig_bias_stagnation(); fig_dro_flips(); fig_risk_heatmap(); fig_fused(); fig_pareto(); fig_timeline()
print('generated',len(list(FIG.glob('*.svg'))),'figures')
