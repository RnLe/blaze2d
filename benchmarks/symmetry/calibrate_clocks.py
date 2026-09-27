#!/usr/bin/env python3
"""Compare independent host/guest clocks during idle and owned CPU intervals."""
import argparse
import json
from pathlib import Path
import time

from quiet_host import QuietHost, assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=3)
    args=p.parse_args();rows=[]
    with QuietHost() as host:
        for _ in range(args.cycles):
            for mode in ['idle','owned']:
                if mode=='idle':
                    row=host.interval()
                else:
                    start=host.snapshot();deadline=time.process_time()+2
                    while time.process_time()<deadline:pass
                    end=host.snapshot();row=dict(start=start,end=end,**assess(start,end))
                rows.append(dict(mode=mode,**row))
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ['start','end']}))


if __name__=='__main__':main()
