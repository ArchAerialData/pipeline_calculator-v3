"""A wrapping, keyboard-operable disclosure without an Aqua button surface.

Tk labels paint their explicit background rather than delegating a button face
to macOS. Keeping normal Tk text layout also preserves wrapping and DPI metrics.
"""
import tkinter as tk


class DisclosureButton(tk.Label):
    def __init__(self, master, *, command, **kwargs):
        self.command = command
        self._normal = kwargs.get('bg', '#272D35')
        self._hover = kwargs.pop('activebackground', '#34404E')
        kwargs.pop('activeforeground', None)
        self._pressed_color = '#1F4665'
        self._inside = False
        self._pressed = False
        self._key = None
        super().__init__(master, disabledforeground='#B6C0CE', **kwargs)
        self.bind('<Enter>', self._enter)
        self.bind('<Leave>', self._leave)
        self.bind('<ButtonPress-1>', self._press)
        self.bind('<ButtonRelease-1>', self._release)
        self.bind('<KeyPress-space>', self._key_press)
        self.bind('<KeyRelease-space>', self._key_release)
        self.bind('<KeyPress-Return>', self._key_press)
        self.bind('<KeyRelease-Return>', self._key_release)
        self.bind('<FocusOut>', self._blur)

    def invoke(self):
        if self.cget('state') != 'disabled':
            return self.command()

    def _paint(self):
        color = self._normal
        if self.cget('state') != 'disabled':
            if self._key or (self._pressed and self._inside):
                color = self._pressed_color
            elif self._inside:
                color = self._hover
        self.configure(bg=color)

    def _enter(self, event):
        self._inside = True
        self._paint()

    def _leave(self, event):
        self._inside = False
        self._paint()

    def _press(self, event):
        if self.cget('state') != 'disabled':
            self.focus_set()
            self._pressed = True
            self._inside = True
            self._paint()
        return 'break'

    def _release(self, event):
        activate = self._pressed and 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        self._pressed = False
        self._paint()
        if activate:
            self.invoke()
        return 'break'

    def _key_press(self, event):
        if self.cget('state') != 'disabled' and self._key != event.keysym:
            self._key = event.keysym
            self._paint()
            if event.keysym == 'Return':
                self.invoke()
        return 'break'

    def _key_release(self, event):
        activate = self._key == event.keysym == 'space'
        self._key = None
        self._paint()
        if activate:
            self.invoke()
        return 'break'

    def _blur(self, event):
        self._key = None
        self._pressed = False
        self._paint()
