from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from pathlib import Path
from threading import Lock
import json, random, math

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

PORT=5000
ROOT=Path(__file__).parent
rng=np.random.default_rng(27)
N=1600

df=pd.DataFrame({
 "claim_id":[f"CLM-{i:05d}" for i in range(1,N+1)],
 "customer_id":[f"CUST-{rng.integers(100,650):04d}" for _ in range(N)],
 "device_id":[f"DEV-{rng.integers(100,900):04d}" for _ in range(N)],
 "repair_center_id":[f"RC-{rng.integers(1,80):03d}" for _ in range(N)],
 "claim_amount":np.round(rng.gamma(2.1,260,size=N)+80,2),
 "customer_claim_count":rng.integers(1,8,size=N),
 "recent_claims":rng.integers(0,5,size=N),
 "previous_fraud_flag":rng.binomial(1,.13,size=N),
 "repair_frequency":rng.integers(1,10,size=N),
 "device_age":rng.integers(1,61,size=N),
 "days_since_purchase":rng.integers(20,900,size=N),
 "entity_risk_count":rng.integers(0,6,size=N)
})
signal=(.0019*df.claim_amount+.16*df.customer_claim_count+.22*df.recent_claims+.95*df.previous_fraud_flag+.11*df.repair_frequency+.025*df.device_age+.08*df.entity_risk_count-rng.normal(2.4,.8,N))
df["fraud_label"]=(signal>np.quantile(signal,.597)).astype(int)
features=["claim_amount","customer_claim_count","recent_claims","previous_fraud_flag","repair_frequency","device_age","days_since_purchase","entity_risk_count"]
X=df[features]; y=df.fraud_label
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=27)
model=RandomForestClassifier(n_estimators=220,max_depth=9,min_samples_leaf=4,class_weight="balanced",random_state=27,n_jobs=-1)
model.fit(Xtr,ytr)
pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,prob),"confusion_matrix":confusion_matrix(yte,pred).tolist()}
df["risk_score"]=model.predict_proba(X)[:,1]
df["risk"]=pd.cut(df.risk_score,[-1,.35,.65,2],labels=["LOW","MEDIUM","HIGH"]).astype(str)
importance=(pd.Series(model.feature_importances_,index=features).sort_values(ascending=False)/model.feature_importances_.sum()*100).round(1)

def row(r):
    factors=[]
    if r.claim_amount>df.claim_amount.quantile(.75): factors.append("Claim amount is above the dataset's upper quartile")
    if r.customer_claim_count>=5: factors.append("Elevated customer claim count")
    if r.recent_claims>=3: factors.append("Multiple recent claims")
    if r.previous_fraud_flag: factors.append("Previous risk indicator is present")
    if r.entity_risk_count>=3: factors.append("Multiple linked risk entities")
    if not factors: factors.append("No single dominant signal; review the combined risk score")
    return {"id":r.claim_id,"customer":r.customer_id,"device":r.device_id,"repair_center":r.repair_center_id,"amount":float(r.claim_amount),"score":float(r.risk_score),"risk":r.risk,"factors":factors,"summary":"Automated triage summary: the claim combines the displayed risk signals into a model score. Human investigation should verify the underlying evidence before any decision."}

def graph(cid):
    r=df[df.claim_id==cid]
    if r.empty:return None
    r=r.iloc[0]; return {"claim_id":cid,"nodes":[{"id":r.customer_id,"type":"Customer","label":r.customer_id},{"id":cid,"type":"Claim","label":cid},{"id":r.device_id,"type":"Device","label":r.device_id},{"id":r.repair_center_id,"type":"Repair Center","label":r.repair_center_id}],"edges":[{"source":r.customer_id,"target":cid},{"source":cid,"target":r.device_id},{"source":cid,"target":r.repair_center_id}]}

class Handler(BaseHTTPRequestHandler):
    def send_json(self,status,obj):
        b=json.dumps(obj).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=="/":
            b=(ROOT/"demo.html").read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        if p=="/api/health": return self.send_json(200,{"status":"ok","model":"RandomForestClassifier","claims":len(df)})
        if p=="/api/model": return self.send_json(200,metrics)
        if p=="/api/summary":
            c=df.risk.value_counts(); q=df.sort_values("risk_score",ascending=False).head(10)
            return self.send_json(200,{"total":len(df),"low":int(c.get("LOW",0)),"medium":int(c.get("MEDIUM",0)),"high":int(c.get("HIGH",0)),"average_risk":float(df.risk_score.mean()),"fraud_rate":float(df.fraud_label.mean()),"signals":[{"name":k.replace("_"," ").title(),"value":float(v)} for k,v in importance.items()],"queue":[row(x) for _,x in q.iterrows()]})
        if p.startswith("/api/claim/"):
            cid=p.rsplit("/",1)[-1].upper(); r=df[df.claim_id==cid]
            return self.send_json(200,row(r.iloc[0])) if not r.empty else self.send_json(404,{"error":"Claim not found"})
        if p.startswith("/api/graph/"):
            g=graph(p.rsplit("/",1)[-1].upper()); return self.send_json(200,g) if g else self.send_json(404,{"error":"Claim not found"})
        self.send_json(404,{"error":"Not found"})
    def log_message(self,*args): pass

if __name__=="__main__":
    print(f"ClaimGuard AI running at http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1",PORT),Handler).serve_forever()
