from urllib.request import urlopen
import json
BASE="http://127.0.0.1:5000"
def get(p):
    with urlopen(BASE+p,timeout=5) as r:
        assert r.status==200
        return json.loads(r.read().decode())
h=get("/api/health"); assert h["status"]=="ok"
s=get("/api/summary"); assert s["total"]==1600
m=get("/api/model"); assert 0<=m["roc_auc"]<=1
c=get("/api/claim/"+s["queue"][0]["id"]); assert c["risk"] in {"LOW","MEDIUM","HIGH"}
g=get("/api/graph/"+c["id"]); assert len(g["nodes"])==4
print("ClaimGuard API smoke test: PASS")
