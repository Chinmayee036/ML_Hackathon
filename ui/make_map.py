"""Person 4: Folium map for one trip. Usage: make_map.py <trip_dir> <out_html> <out_mock_json>"""
import sys, os, json
import numpy as np, folium

trip_dir, out_html, out_json = sys.argv[1:4]
tid = os.path.basename(trip_dir.rstrip("/"))
p = os.path.join(trip_dir, tid)
trk = np.loadtxt(p + ".track"); nodes = np.loadtxt(p + ".nodes")
arcs = np.loadtxt(p + ".arcs", dtype=int); route = np.loadtxt(p + ".route", dtype=int, ndmin=1)

# local metre projection (equirectangular around trip centre)
lon0, lat0 = trk[:, 0].mean(), trk[:, 1].mean()
kx = 111320 * np.cos(np.radians(lat0)); ky = 110540
def xy(lon, lat): return np.c_[(lon - lon0) * kx, (lat - lat0) * ky]
P = xy(trk[:, 0], trk[:, 1]); N = xy(nodes[:, 0], nodes[:, 1])
A, B = N[arcs[:, 0]], N[arcs[:, 1]]

# restrict to arcs near the trip (bbox + 300 m) so it stays fast
lo, hi = P.min(0) - 300, P.max(0) + 300
keep = np.where(~((np.maximum(A, B) < lo).any(1) | (np.minimum(A, B) > hi).any(1)))[0]
Ak, Bk = A[keep], B[keep]; AB = Bk - Ak; L2 = np.maximum((AB ** 2).sum(1), 1e-9)

def nearest(pt):
    t = np.clip(((pt - Ak) * AB).sum(1) / L2, 0, 1)
    d = np.linalg.norm(Ak + t[:, None] * AB - pt, axis=1)
    return keep[d.argmin()], d.min()
best = [nearest(pt) for pt in P]
base_arc = np.array([b[0] for b in best]); base_d = np.array([b[1] for b in best])

ref = set(route.tolist())
und = lambda a: tuple(sorted(arcs[a]))
ref_und = {und(a) for a in ref}
exact = np.mean([a in ref for a in base_arc])
undirected = np.mean([und(a) in ref_und for a in base_arc])
wrong = np.where([und(a) not in ref_und for a in base_arc])[0]
print(f"{tid}: {len(trk)} GPS pts, {len(keep)} nearby arcs, route {len(route)} arcs")
print(f"nearest-road baseline: exact-arc match {exact:.3f}, ignoring direction {undirected:.3f}, mean dist {base_d.mean():.1f} m")

def ll(node_ids): return [[nodes[n, 1], nodes[n, 0]] for n in node_ids]
def arc_line(a): return ll(arcs[a])

m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles=None)
folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                 attr="Tiles &copy; Esri", name="Esri street map", max_zoom=19).add_to(m)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap (needs http:// serving)", show=False).add_to(m)
fg_net = folium.FeatureGroup("Nearby roads (network)", show=False)
near_set = set(base_arc.tolist())
for a in keep[:: 1]:
    if np.min(np.linalg.norm(P[::10] - (A[a] + B[a]) / 2, axis=1)) < 120:
        folium.PolyLine(arc_line(a), color="#999", weight=1, opacity=.6).add_to(fg_net)
fg_ref = folium.FeatureGroup("Reference route (ground truth)")
for a in route: folium.PolyLine(arc_line(a), color="#1a9850", weight=6, opacity=.8).add_to(fg_ref)
fg_base = folium.FeatureGroup("Baseline: nearest road per GPS point")
for a in sorted(near_set): folium.PolyLine(arc_line(a), color="#e08214", weight=3, opacity=.9).add_to(fg_base)
fg_gps = folium.FeatureGroup("GPS points")
for i in range(0, len(trk), 3):
    folium.CircleMarker([trk[i, 1], trk[i, 0]], radius=2, color="#2166ac", fill=True, weight=1,
                        tooltip=f"#{i} t={trk[i,2]:.0f}s nearest={base_d[i]:.1f} m").add_to(fg_gps)
fg_wrong = folium.FeatureGroup("Baseline mistakes (nearest road not on route)")
for i in wrong[::max(1, len(wrong)//300)]:
    folium.CircleMarker([trk[i, 1], trk[i, 0]], radius=4, color="#d73027", fill=True,
                        tooltip=f"#{i}: nearest arc {base_arc[i]} not on route").add_to(fg_wrong)
# --- highlight the longest run of consecutive baseline mistakes (the "spot where nearest-road goes wrong")
runs=[];start=None
for i in range(len(trk)):
    w=und(base_arc[i]) not in ref_und
    if w and start is None: start=i
    if (not w or i==len(trk)-1) and start is not None:
        runs.append((start,i-1 if not w else i)); start=None
fg_spot = folium.FeatureGroup("WHERE NEAREST-ROAD FAILS (worst run)")
spot=None
if runs:
    a,b=max(runs,key=lambda r:r[1]-r[0]); spot=(int(a),int(b))
    mid=(a+b)//2
    folium.Circle([trk[mid,1],trk[mid,0]],radius=60,color="#d73027",fill=False,weight=3).add_to(fg_spot)
    folium.Marker([trk[mid,1],trk[mid,0]],icon=folium.Icon(color="red",icon="exclamation-sign"),
      popup=folium.Popup(f"<b>Nearest-road fails here</b><br>GPS #{a}-#{b} ({b-a+1} points in a row) snap to roads NOT on the reference route.<br><br><b>What the HMM uses to fix it:</b> a candidate on the wrong road is only reachable from the previous road via a long detour, so the transition score is low; Viterbi prefers the connected path along the real route.",max_width=330)).add_to(fg_spot)
    print("worst run: GPS points",a,"to",b,f"({b-a+1} pts) at lat {trk[mid,1]:.5f}, lon {trk[mid,0]:.5f}")
for g in (fg_net, fg_ref, fg_base, fg_gps, fg_wrong, fg_spot): g.add_to(m)
legend='<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif"><b>Trip %s</b><br><span style="color:#1a9850">&#9632;</span> Reference route<br><span style="color:#e08214">&#9632;</span> Nearest-road baseline<br><span style="color:#2166ac">&#9679;</span> GPS points<br><span style="color:#d73027">&#9679;</span> Baseline mistakes<br><span style="color:#999">&#9632;</span> Road network</div>'%tid
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m); m.save(out_html)

# mock API response (agree shape with Person 3)
mock = {"trip_id": tid,
        "gps_points": [{"idx": i, "lon": float(trk[i, 0]), "lat": float(trk[i, 1]), "t": float(trk[i, 2])} for i in range(0, len(trk), 20)],
        "predicted_route": {"arc_ids": [int(a) for a in route[:5]],
                            "geometry": [[float(x), float(y)] for a in route[:5] for x, y in ([nodes[arcs[a, 0]][:2], nodes[arcs[a, 1]][:2]])]},
        "metrics": {"classifier_only_top1": float(undirected), "hmm_top1": None, "route_continuity": None, "n_points": int(len(trk))},
        "worst_baseline_run": spot, "note": "MOCK: predicted_route is the reference route's first 5 arcs; hmm_* filled by Person 3's API"}
json.dump(mock, open(out_json, "w"), indent=1)
