"""Generate a scalable diagram of the actual implementation; no model results are invented."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'report_support/figures';out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(10,9));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
def box(x,y,w,h,text,color='#edf3fa'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=color,ec='#31516c',lw=1.1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=13,color='#31516c',lw=1.2))
box(.4,8.75,4.1,.65,'[CLS] goal [SEP] candidate 1 [SEP]')
box(5.5,8.75,4.1,.65,'[CLS] goal [SEP] candidate 2 [SEP]')
box(1.7,7.45,6.6,.85,'Shared embeddings: token + position + segment + difference tag\nBatch: B x 2 x L x 256; L <= 139')
arrow(2.45,8.67,3.5,8.38);arrow(7.55,8.67,6.5,8.38)
box(1.7,6.1,6.6,.85,'Shared pre-LN Transformer encoder: 4 blocks\n4 attention heads; 64 features/head; FFN width 1024')
arrow(5,7.36,5,7.02)
box(1.7,4.75,6.6,.85,'Cross-solution attention in both directions\nQueries: own option; keys/values: rival solution + CLS')
arrow(5,6.01,5,5.67)
box(1.7,3.4,6.6,.85,'Comparison [q; c; q-c; q*c]: 1024 features\nFusion MLP + residual + LayerNorm: 256 features')
arrow(5,4.66,5,4.32)
box(.5,1.95,5.7,.85,'Additive attention over solution tokens\nLearned difference-tag bias; pooled width 256','#e8f2ea')
box(7,1.95,2.5,.85,'Updated CLS\nwidth 256','#e8f2ea')
arrow(3.6,3.31,3.3,2.87);arrow(7.5,3.31,8.2,2.87)
box(1.7,.45,6.6,.85,'Concatenate pooled + CLS: 512\nShared scorer 512 -> 256 -> 1 per option; softmax(s1,s2)')
arrow(3.35,1.86,4.0,1.37);arrow(8.25,1.86,6.5,1.37)
ax.set_title('DACT: Difference-Aware Contrastive Transformer',fontsize=15,pad=12)
fig.text(.5,.015,'Goal-candidate interaction occurs inside each encoder. MLM warm-up uses the encoder only (not shown).',ha='center',fontsize=9)
fig.savefig(out/'architecture.svg',bbox_inches='tight')
fig.savefig(out/'architecture.png',dpi=300,bbox_inches='tight')
plt.close(fig)
print(out/'architecture.svg')
