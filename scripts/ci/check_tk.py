"""Verify the installed Tcl/Tk runtime in an isolated, bounded process."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.validation.gui_process import run_gui


def main():
    probe = """
import sys, tkinter as tk
print('Python:', sys.executable, flush=True)
root = tk.Tk()
root.withdraw()
print('Tcl library:', root.tk.eval('info library'), flush=True)
print('Tcl version:', root.tk.eval('info patchlevel'), flush=True)
print('Tk version:', root.tk.call('package', 'require', 'Tk'), flush=True)
print('Screen:', root.winfo_screenwidth(), root.winfo_screenheight(), flush=True)
root.update_idletasks()
root.destroy()
"""
    result = run_gui([sys.executable, '-u', '-c', probe], timeout=20)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    result.check_returncode()


if __name__ == '__main__':
    main()
