"""Run file work outside Tk while a small progress window stays responsive."""
import threading

import customtkinter as ctk

from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.window import fit_window


def run_background_action(parent, title, message, action):
    """Return (result, error), or (None, None) if the application was closed.

    Only plain Python values cross the thread boundary. Closing the parent
    detaches polling; completed workers never address a destroyed Tk widget.
    """
    window = ctk.CTkToplevel(parent)
    window.title(title)
    window.transient(parent)
    WrappedLabel(window, text=message, justify='left').pack(fill='x', padx=24, pady=24)
    progress = ctk.CTkProgressBar(window, mode='indeterminate')
    progress.pack(fill='x', padx=24, pady=(0, 24))
    state = {'result': None, 'error': None, 'closed': False, 'poll': None, 'finished': False}
    done = threading.Event()

    def detached(event):
        if event.widget is window:
            state['closed'] = True
            if state['poll'] is not None:
                window.after_cancel(state['poll'])
                state['poll'] = None

    def worker():
        try:
            state['result'] = action()
        except Exception as error:
            state['error'] = error
        finally:
            done.set()

    def poll():
        state['poll'] = None
        if state['closed']:
            return
        if done.is_set():
            state['finished'] = True
            progress.stop()
            window.destroy()
        else:
            state['poll'] = window.after(50, poll)

    window.bind('<Destroy>', detached, add='+')
    # A partially written workbook cannot be safely cancelled mid-save.
    window.protocol('WM_DELETE_WINDOW', lambda: None)
    fit_window(window, (480, 180), parent=parent)
    window.wait_visibility()
    window.grab_set()
    progress.start()
    try:
        threading.Thread(target=worker, daemon=True).start()
    except RuntimeError as error:
        state['error'] = error
        done.set()
    poll()
    if window.winfo_exists():
        window.wait_window()
    return (state['result'], state['error']) if state['finished'] else (None, None)
