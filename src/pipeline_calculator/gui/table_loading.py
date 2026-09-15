"""Time-bounded table population, paused while hidden and cancelled on destroy."""
import logging
import time
import tkinter as tk


class TableLoader:
    def __init__(self, tree, rows):
        self.tree = tree
        self.rows = iter(rows)
        self.pending = None
        self.count = 0
        self.status = tk.Label(tree.master, text='Loading rows…', bg='#242424', fg='#F1F4F8')
        self.status.grid(row=2, column=0, columnspan=2, sticky='ew')
        self.map_binding = tree.bind('<Map>', self._schedule, add='+')
        tree.bind('<Unmap>', self._pause, add='+')
        tree.bind('<Destroy>', self._destroy, add='+')
        # Small tables retain immediate construction. Large tables yield before
        # they can block navigation, and resume only when their page is visible.
        self._step()

    def _schedule(self, event=None):
        if self.rows is not None and self.pending is None and self.tree.winfo_viewable():
            self.pending = self.tree.after(1, self._step)

    def _pause(self, event=None):
        if self.pending is not None:
            self.tree.after_cancel(self.pending)
            self.pending = None

    def _destroy(self, event):
        if event.widget is self.tree:
            self._pause()
            if self.rows is not None and hasattr(self.rows, 'close'):
                self.rows.close()
            self.rows = None

    def _step(self):
        self.pending = None
        started = time.perf_counter()
        try:
            for _ in range(200):
                next(self.rows)
                self.count += 1
                if time.perf_counter() - started >= .008:
                    break
        except StopIteration:
            self.rows = None
            self.status.destroy()
            self.status = None
            return
        except Exception:
            logging.getLogger(__name__).exception('Could not populate result table')
            self.rows = None
            self.status.configure(text='Could not load all rows. Select another scope or reimport to retry; export remains available.')
            return
        self.status.configure(text=f'Loading rows… {self.count:,} loaded')
        self._schedule()


def load_rows(tree, rows):
    tree.row_loader = TableLoader(tree, rows)
