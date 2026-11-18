"""Folium map of one trip: GPS, nearby roads, reference route, nearest-road baseline,
baseline errors, and GPS->aligned-reference links.
Usage: python scripts/03_folium_map.py --root "path/to/Student folder" --partition train --trip 00000024
"""
import argparse, os, sys
import numpy as np, folium
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.io import trip_dir
from pathsnap.candidates import build_trip

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--partition", default="train")
ap.add_argument("--trip", default="00000024"); ap.add_argument("--out", default="outputs/maps")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

df, info, net, tr = build_trip(trip_dir(a.root, a.partition, a.trip))
inv = net.proj.inv
def line(arc):  # arc id -> [(lat, lon), (lat, lon)]
    lo, la = inv(np.stack([net.A[arc], net.B[arc]]))
    return [(la[0], lo[0]), (la[1], lo[1])]

obs = df.drop_duplicates("obs_idx").set_index("obs_idx")
top = df[df.dist_rank == 0].set_index("obs_idx")
wrong = top[top.label == 0]
lat0, lon0 = obs.lat.mean(), obs.lon.mean()
# centre on the first nearest-road error if there is one (interesting spot), else on the trip
if len(wrong):
    w = wrong.iloc[len(wrong) // 3]; centre, zoom = (w.lat, w.lon), 18
else:
    centre, zoom = (lat0, lon0), 15
m = folium.Map(location=centre, zoom_start=zoom, tiles="OpenStreetMap")

# nearby roads: all arcs that were a candidate for any fix (dedupe direction)
fg_road = folium.FeatureGroup("Candidate roads (grey)", show=True)
seen = set()
for arc in df.arc_id.unique():
    key = tuple(sorted(net.arcs[arc]))
    if key in seen: continue
    seen.add(key); folium.PolyLine(line(arc), color="gray", weight=2, opacity=.6).add_to(fg_road)
fg_road.add_to(m)

fg_ref = folium.FeatureGroup("Reference route (green)", show=True)
for arc in tr["route"]:
    folium.PolyLine(line(arc), color="green", weight=5, opacity=.8, tooltip=f"ref arc {arc}").add_to(fg_ref)
fg_ref.add_to(m)

fg_nn = folium.FeatureGroup("Nearest-road baseline (orange)", show=True)
for arc in top.arc_id.unique():
    folium.PolyLine(line(arc), color="orange", weight=3, opacity=.9, dash_array="6", tooltip=f"nearest arc {arc}").add_to(fg_nn)
fg_nn.add_to(m)

fg_gps = folium.FeatureGroup("GPS fixes (blue)", show=True)
for i, r in obs.iterrows():
    folium.CircleMarker((r.lat, r.lon), radius=2.5, color="blue", fill=True, weight=1,
                        tooltip=f"obs {i}  t={r.t:.0f}s  nearest={int(top.loc[i].arc_id)}  ref={int(r.ref_arc)}").add_to(fg_gps)
fg_gps.add_to(m)

fg_err = folium.FeatureGroup("Nearest-road errors (red)", show=True)
for i, r in wrong.iterrows():
    folium.CircleMarker((r.lat, r.lon), radius=5, color="red", fill=True, fill_opacity=.8,
                        tooltip=f"obs {i}: nearest arc {int(r.arc_id)} != ref arc {int(r.ref_arc)}").add_to(fg_err)
fg_err.add_to(m)

# link each fix to its aligned reference arc (visual check of label alignment)
fg_al = folium.FeatureGroup("GPS -> aligned reference arc links (purple)", show=False)
from pathsnap.geo import point_segment
P = obs[["x", "y"]].values; ra = obs.ref_arc.values.astype(int)
_, _, foot = point_segment(P, net.A[ra], net.B[ra])
flo, fla = inv(foot)
for k, i in enumerate(obs.index):
    folium.PolyLine([(obs.lat.iloc[k], obs.lon.iloc[k]), (fla[k], flo[k])], color="purple", weight=1).add_to(fg_al)
fg_al.add_to(m)

legend = f"""<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;
border:1px solid #888;font-size:12px"><b>Trip {a.trip} ({a.partition})</b><br>
<span style="color:blue">&#9679;</span> GPS fixes<br><span style="color:green">&#9632;</span> reference route<br>
<span style="color:orange">&#9632;</span> nearest-road baseline<br><span style="color:red">&#9679;</span> nearest-road error<br>
<span style="color:gray">&#9632;</span> candidate roads<br>
nearest-road acc: {info.get('nearest_road_acc', float('nan')):.3f}<br>candidate miss rate: {info.get('candidate_miss_rate', float('nan')):.3f}</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m)
out = f"{a.out}/trip_{a.trip}_{a.partition}.html"; m.save(out); print("saved", out, {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()})
