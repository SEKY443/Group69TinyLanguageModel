"""Run B0/B1 on the unchanged historical split; retain model/prediction evidence separately."""
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import Config,environment_info
from data import load_piqa
from baselines import majority_baseline,tfidf_lr_baseline,tfidf_lr_predict
from evaluate import accuracy,save_predictions

def main():
    out=ROOT/'experiments/completed/outputs_replication_baselines'
    if out.exists():raise FileExistsError(out)
    (out/'predictions').mkdir(parents=True)
    tr,va,te=load_piqa(Config(data_dir=str(ROOT/'../../PIQA')))
    start=time.time()
    fitted=tfidf_lr_baseline(tr,va)
    joblib.dump({'vectorizer':fitted['vectorizer'],'classifier':fitted['classifier']},out/'B1.joblib')
    results={}
    for name in ('B0','B1'):
        results[name]={}
        for split,rows in (('val',va),('test',te)):
            if name=='B0':prediction={'pred':majority_baseline(tr,rows)}
            else:
                vec=fitted['vectorizer'];x=vec.transform([r['sol2'] for r in rows])-vec.transform([r['sol1'] for r in rows])
                prediction={'pred':fitted['classifier'].predict(x),'prob':fitted['classifier'].predict_proba(x)[:,1]}
            save_predictions(str(out/'predictions'/f'{name}_{split}.jsonl'),rows,prediction,name,split)
            results[name][split+'_accuracy']=accuracy(prediction['pred'],[r['label'] for r in rows])
    results['B1']['selected_C']=fitted['C']
    results['environment']=environment_info('cpu')
    results['seconds']=time.time()-start
    results['protocol']='Fixed historical split. C selected on validation before any test inference. Historical outputs unchanged.'
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
