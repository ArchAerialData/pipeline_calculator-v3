"""Measure real worker cancellation acknowledgement at requested processing stages."""
import argparse
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import environment, parallel, save_json, write_fixture
from pipeline_calculator.gui.controllers.analysis_controller import AnalysisJob
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.core.execution import ExecutionContext


def run(output):
    path=write_fixture(output/'dense.kml',parallel((0,2,4,6),length=25000))
    rows=[]
    stages=['Reading documents','Calculating source lengths','Segmenting path','Building spatial index',
            'Searching neighbors','Qualifying sections','Building corridors','Building group graph',
            'Sorting group candidates','Calculating savings']
    for target in stages:
        entered=threading.Event()
        class ObservedContext(ExecutionContext):
            def report(self,stage,*args):
                super().report(stage,*args)
                if stage==target:
                    entered.set()
        job=AnalysisJob(str(path),AnalysisParameters())
        job.context=ObservedContext(job.job_id)
        job.start()
        if not entered.wait(30):
            job.cancel()
            if not job.done.wait(30):
                raise RuntimeError('Worker failed to acknowledge cancellation')
            raise RuntimeError(f'Workload did not reach requested stage {target}')
        snapshot=job.context.snapshot()
        start=time.perf_counter()
        job.cancel()
        if not job.done.wait(30):
            raise RuntimeError('Worker failed to acknowledge cancellation')
        elapsed=time.perf_counter()-start
        rows.append({'requested_stage':target,'observed_stage':snapshot.stage,'seconds':elapsed,
                     'state':job.state,'partial_result':job.result is not None})
        job._thread.join(5)
    save_json(output/'report.json',{'environment':environment(),'measurements':rows,
        'limit':'Single runs; a requested stage may advance before the UI thread runs. Not a hard upper latency bound.'})
    print(f'{len(rows)} cancellations; max acknowledgement {max(r["seconds"] for r in rows):.6f} s')
    return int(any(r['state']!='cancelled' or r['partial_result'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run(args.output))
