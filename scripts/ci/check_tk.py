"""Verify the installed Tcl/Tk runtime in an isolated, bounded process."""
import subprocess
import sys


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
    subprocess.run([sys.executable, '-u', '-c', probe], check=True, timeout=20)


if __name__ == '__main__':
    main()
