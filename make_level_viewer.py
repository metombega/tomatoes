"""
Build a standalone HTML viewer (three.js, opens in any browser) for the
per-level clouds written by merge_stacked_levels.py (level_<name>.ply), so
the levels can be switched on/off, recolored by level and sliced to inspect
how well they match.

Usage:
    python make_level_viewer.py [merged_folder] [output_html]

Defaults: merged_folder=rgbd_photos/top_and_bottom_view_output/merged,
output_html=<merged_folder>/level_viewer.html
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

MERGED_FOLDER = "rgbd_photos/top_and_bottom_view_output/merged"
MAX_POINTS_PER_LEVEL = 80000
LEVEL_COLORS = ["#2f7de1", "#f08a24", "#3bb273", "#c04cd8"]

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Level viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#111;color:#ddd;font:13px system-ui,sans-serif;overflow:hidden}
#ui{position:absolute;top:10px;left:10px;background:#000a;padding:10px 12px;border-radius:8px;max-width:260px}
#ui h4{margin:8px 0 4px;font-size:12px;color:#9ab}
#ui label{display:block;margin:2px 0}
#ui input[type=range]{width:100%}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
#info{color:#9ab;margin-top:6px}
</style></head><body>
<div id="ui">
<h4>Levels</h4><div id="levels"></div>
<h4>Color</h4>
<label><input type="radio" name="cm" value="orig" checked> original</label>
<label><input type="radio" name="cm" value="level"> by level</label>
<h4>Point size <span id="psv"></span></h4><input id="ps" type="range" min="0.5" max="6" step="0.25" value="2">
<h4>Slice along
<select id="ax"><option>X</option><option selected>Y</option><option>Z</option></select></h4>
<label>from <input id="lo" type="range" min="0" max="1000" value="0"></label>
<label>to <input id="hi" type="range" min="0" max="1000" value="1000"></label>
<div id="info"></div>
<div id="info2" style="color:#789;margin-top:4px">drag: rotate · wheel: zoom · right-drag: pan</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.001,10);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setSize(innerWidth,innerHeight);ren.setClearColor(0x111111);
document.body.appendChild(ren.domElement);
const ctl=new THREE.OrbitControls(cam,ren.domElement);
const b64=s=>{const bin=atob(s),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;};
const mid=DATA.center, lv=[];
DATA.levels.forEach((L,k)=>{
  const pos=new Float32Array(b64(L.pos).buffer), col=new Uint8Array(b64(L.col).buffer), n=pos.length/3;
  const lc=new THREE.Color(L.color), colOrig=new Float32Array(n*3), colLvl=new Float32Array(n*3);
  for(let i=0;i<n;i++){colOrig[3*i]=col[3*i]/255;colOrig[3*i+1]=col[3*i+1]/255;colOrig[3*i+2]=col[3*i+2]/255;colLvl[3*i]=lc.r;colLvl[3*i+1]=lc.g;colLvl[3*i+2]=lc.b;}
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',new THREE.BufferAttribute(colOrig.slice(),3));
  const mat=new THREE.PointsMaterial({size:2,sizeAttenuation:false,vertexColors:true});
  const pts=new THREE.Points(g,mat);scene.add(pts);
  lv.push({L,n,pos,g,mat,pts,colOrig,colLvl,vis:true});
  const row=document.createElement('label');
  row.innerHTML=`<input type="checkbox" checked> <span class="sw" style="background:${L.color}"></span>${L.name} (${n})`;
  row.firstChild.onchange=e=>{lv[k].vis=e.target.checked;update();};
  document.getElementById('levels').appendChild(row);
});
const ext=DATA.extent; cam.position.set(0,0,ext*3); ctl.target.set(0,0,0);
const ax=document.getElementById('ax'),lo=document.getElementById('lo'),hi=document.getElementById('hi');
function update(){
  const a=ax.selectedIndex, mn=DATA.min[a], mx=DATA.max[a];
  let f0=lo.value/1000,f1=hi.value/1000; if(f0>f1)[f0,f1]=[f1,f0];
  const v0=mn+(mx-mn)*f0, v1=mn+(mx-mn)*f1, cm=document.querySelector('input[name=cm]:checked').value;
  let shown=0;
  lv.forEach(o=>{
    o.pts.visible=o.vis; o.mat.size=+document.getElementById('ps').value;
    const c=o.g.attributes.color.array, src=cm==='orig'?o.colOrig:o.colLvl;
    for(let i=0;i<o.n;i++){
      const inside=o.pos[3*i+a]>=v0&&o.pos[3*i+a]<=v1;
      const k=3*i; if(inside){c[k]=src[k];c[k+1]=src[k+1];c[k+2]=src[k+2];shown++;}else{c[k]=c[k+1]=c[k+2]=0;}
    }
    o.g.attributes.color.needsUpdate=true;
  });
  document.getElementById('psv').textContent=document.getElementById('ps').value;
  document.getElementById('info').textContent=`slice ${(v0*1000).toFixed(1)} … ${(v1*1000).toFixed(1)} mm`;
}
// hidden-by-slice points are painted black (invisible on the black-ish bg) -- drop them for clarity:
ren.setClearColor(0x000000);
[ax,lo,hi,document.getElementById('ps'),...document.querySelectorAll('input[name=cm]')].forEach(e=>e.oninput=update);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
update();(function loop(){requestAnimationFrame(loop);ctl.update();ren.render(scene,cam);})();
</script></body></html>
"""


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(MERGED_FOLDER)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "level_viewer.html"
    files = sorted(folder.glob("level_*.ply"))
    if not files:
        raise SystemExit(f"No level_*.ply in {folder} -- run merge_stacked_levels.py first")

    clouds = [o3d.io.read_point_cloud(str(f)) for f in files]
    allp = np.vstack([np.asarray(c.points) for c in clouds])
    center = (allp.min(axis=0) + allp.max(axis=0)) / 2
    rng = np.random.default_rng(0)

    levels = []
    for k, (f, c) in enumerate(zip(files, clouds)):
        p, col = np.asarray(c.points), np.asarray(c.colors)
        if len(p) > MAX_POINTS_PER_LEVEL:
            idx = rng.choice(len(p), MAX_POINTS_PER_LEVEL, replace=False)
            p, col = p[idx], col[idx]
        levels.append({
            "name": f.stem[len("level_"):],
            "color": LEVEL_COLORS[k % len(LEVEL_COLORS)],
            "pos": base64.b64encode((p - center).astype(np.float32).tobytes()).decode(),
            "col": base64.b64encode((np.clip(col, 0, 1) * 255).astype(np.uint8).tobytes()).decode(),
        })
    data = {"levels": levels, "center": center.tolist(),
            "min": (allp.min(axis=0) - center).tolist(), "max": (allp.max(axis=0) - center).tolist(),
            "extent": float(np.ptp(allp, axis=0).max())}
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"Viewer -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
