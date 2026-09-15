"""Own external widget subscriptions without removing other Tk handlers."""
import tkinter as tk

import customtkinter as ctk


def _detach(owner, target, token):
    """Remove our generated Tk callback and its additive blank separator."""
    lines = owner.tk.call(*target).splitlines(keepends=True)
    prefix = f'if {{"[{token} '
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            del lines[index]
            if index and lines[index - 1] == '\n':
                del lines[index - 1]
            break
    owner.tk.call(*target, ''.join(lines))
    owner.deletecommand(token)


class ConfigureBinding:
    """CTkFrame forwards Configure to its canvas but drops the binding ID.

    Bind to that native canvas directly so a child can detach independently of
    its parent. Other supported parents (including scroll frames) use Tk bind.
    """
    def __init__(self, parent, callback):
        self.target = parent._canvas if isinstance(parent, ctk.CTkFrame) else parent
        self.token = tk.Misc.bind(self.target, '<Configure>', callback, add='+')

    def close(self):
        if self.token is not None:
            if self.target.winfo_exists():
                _detach(self.target, ('bind', self.target._w, '<Configure>'), self.token)
            self.token = None


class OwnedGlobalBindings:
    """Selectively remove CTk 5.2.2 scroll-frame bindings from the Tk root."""
    def __init__(self):
        self.bindings = []

    def add(self, widget, sequence, func, add):
        root = widget._root()
        token = root.bind_all(sequence, func, add)
        if token and func:
            self.bindings.append((root, sequence, token))
        return token

    def close(self):
        for root, sequence, token in self.bindings:
            _detach(root, ('bind', 'all', sequence), token)
        self.bindings.clear()
