"""Shared launch routing for both GUIs, with an optional status dialog."""
import threading
import tkinter as tk
from tkinter import messagebox

from pipeline_calculator.gui import config
from pipeline_calculator.gui.actions import open_kml_action
from pipeline_calculator.gui.corridor_presentation import (
    corridor_is_omitted, corridor_unavailable_reason, MAP_OMISSION_NOTE,
)


def launch_corridor(root, section, index):
    if corridor_is_omitted(section):
        raise ValueError(f'This corridor map is unavailable. {corridor_unavailable_reason(section)} '
                         f'{MAP_OMISSION_NOTE}')
    if config.SHOW_CORRIDOR_LAUNCH_DIALOG:
        from pipeline_calculator.gui.dialogs.corridor_dialog import CorridorDialog
        return CorridorDialog(root, section, index)
    return DirectCorridorLaunch(root, section, index)


class DirectCorridorLaunch:
    """Launch without a success modal; report genuine failures on the Tk thread."""
    def __init__(self, root, section, index):
        self.root = root
        self.done = threading.Event()
        self.outcome = None
        self.error = None
        self.closed = False
        self.poll_id = None
        self.destroy_binding = root.bind('<Destroy>', self._destroyed, add='+')

        def worker():
            try:
                self.outcome = open_kml_action.create_and_launch_corridor(section, index)
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.done.set()
        try:
            threading.Thread(target=worker, daemon=True).start()
        except RuntimeError:
            self.close()
            raise
        self._poll()

    def _poll(self):
        self.poll_id = None
        if self.closed:
            return
        if not self.done.is_set():
            self.poll_id = self.root.after(100, self._poll)
            return
        self.close()
        error = self.error
        if not error and self.outcome.status != 'requested':
            error = f'{self.outcome.error}\n\nCorridor KML saved at:\n{self.outcome.path}'
        if error:
            messagebox.showerror('Could not open corridor', error, parent=self.root)

    def _destroyed(self, event):
        if event.widget is self.root:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None
        if self.destroy_binding:
            try:
                self.root.unbind('<Destroy>', self.destroy_binding)
            except tk.TclError:
                pass
            self.destroy_binding = None
