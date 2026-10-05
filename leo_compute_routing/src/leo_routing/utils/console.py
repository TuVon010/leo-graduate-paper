"""Live stdout/stderr transcripts without requiring a terminal or extra packages."""
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import traceback


def console_log_path(output, kind):
    output = Path(output).resolve()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return output.parent / "console_logs" / ("%s_%s_%s_%s.log" % (kind, output.name, stamp, os.getpid()))


class TeeStream:
    def __init__(self, terminal, transcript, lock):
        self.terminal, self.transcript, self.lock = terminal, transcript, lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.transcript.write(text)
            self.terminal.flush()
            self.transcript.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.transcript.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def capture_console(path):
    """Save exceptions before restoring streams; the CLI still exits nonzero."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_out, previous_err = sys.stdout, sys.stderr
    with path.open("x", encoding="utf-8") as transcript:
        lock = threading.RLock()
        sys.stdout = TeeStream(previous_out, transcript, lock)
        sys.stderr = TeeStream(previous_err, transcript, lock)
        try:
            print("[CONSOLE] started=%s | pid=%s | log=%s" %
                  (datetime.now().astimezone().isoformat(timespec="seconds"), os.getpid(), path), flush=True)
            yield path
        except BaseException:
            # The outer interpreter will display the traceback on the restored
            # stderr. Save it here as well, without printing it twice live.
            traceback.print_exc(file=transcript)
            transcript.flush()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = previous_out, previous_err
