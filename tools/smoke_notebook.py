"""Execute the generated notebook in a fresh kernel; development only, no pretrained downloads."""
import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
os.environ['PIQA_SMOKE']='1'
os.environ['PIQA_DATA_DIR']=str((ROOT/'../../PIQA').resolve())
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
nb=nbformat.read(ROOT/'notebooks/CITS4012_69_reproducible.ipynb',as_version=4)
client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    out=ROOT/'evidence'/('CITS4012_69_development_smoke_'+time.strftime('%Y%m%d_%H%M%S')+'.ipynb')
    nbformat.write(nb,out)
print('Executed development notebook:',out)
print('Errors:',sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs))
