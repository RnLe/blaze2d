"""Optional terminal progress driven by actual completed jobs."""


class TerminalProgress:
    """Context manager and callback for run, stream, and checkpointed studies.

    Install blaze2d[progress]. Use with TerminalProgress() as progress, then
    pass progress=progress to the study function. Output is written to stderr.
    """
    def __init__(self):
        try:
            from rich.console import Console
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        except ImportError as error:
            raise RuntimeError("Install blaze2d[progress] for terminal progress") from error
        self.display = Progress(TextColumn("{task.description}"), BarColumn(),
                                TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                                console=Console(stderr=True))
        self.task = None

    def __enter__(self):
        self.display.start()
        return self

    def __exit__(self, *args):
        self.display.stop()

    def __call__(self, event):
        if event['event'] == 'run_start':
            self.task = self.display.add_task('Calculating', total=event['jobs'])
        elif self.task is not None and event['event'] in ('result', 'job_failure'):
            self.display.advance(self.task)
        elif self.task is not None and event['event'] == 'terminal':
            self.display.update(self.task, description=event['status'].replace('_', ' '))
