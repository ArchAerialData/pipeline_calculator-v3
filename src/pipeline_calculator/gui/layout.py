"""Responsive text, actions and parameter fields shared by both entrypoints."""
import customtkinter as ctk


class WrappedLabel(ctk.CTkLabel):
    def __init__(self, master, *, wrap_padding=32, **kwargs):
        self._wrap_padding = wrap_padding
        kwargs.setdefault('width', 1)
        kwargs.setdefault('wraplength', 380)
        super().__init__(master, **kwargs)
        self._wrap_id = None
        self._wrap_width = None
        master.bind('<Configure>', self._wrap, add='+')

    def _wrap(self, event):
        if self.winfo_exists():
            scale = ctk.ScalingTracker.get_widget_scaling(self)
            width = max(40, event.width / scale - self._wrap_padding)
            if self._wrap_width == width:
                return
            self._wrap_width = width
            if self._wrap_id is not None:
                self.after_cancel(self._wrap_id)
            self._wrap_id = self.after(20, self._apply_wrap)

    def _apply_wrap(self):
        self._wrap_id = None
        self.configure(wraplength=self._wrap_width)

    def destroy(self):
        if self._wrap_id is not None:
            self.after_cancel(self._wrap_id)
        super().destroy()


class ActionBar(ctk.CTkFrame):
    def __init__(self, master, actions):
        super().__init__(master, fg_color='transparent')
        self.buttons = [ctk.CTkButton(self, text=text, command=command, width=150, height=34)
                        for text, command in actions]
        self._columns = None
        self.bind('<Configure>', self._arrange, add='+')
        self._arrange()

    def _arrange(self, event=None):
        if not self.buttons:
            return
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        width = event.width if event else 1
        required = 172*scale
        columns = max(1, min(len(self.buttons), int(width/required)))
        if columns == self._columns:
            return
        self._columns = columns
        for i in range(len(self.buttons)):
            self.grid_columnconfigure(i, weight=int(i < columns), uniform='actions' if i < columns else '')
        for i, button in enumerate(self.buttons):
            button.grid(row=i//columns, column=i%columns, sticky='ew', padx=5, pady=4)


def parameter_fields(parent, variables, *, compact=False):
    fields = [
        ('Detection Range (m)', variables[0], 'Maximum centerline separation to bundle'),
        ('Segment Length (m)', variables[1], 'Smaller values give finer estimates and take longer'),
        ('Min Parallel Length (m)', variables[2], 'Minimum continuous bundled section'),
        ('Angular Tolerance (°)', variables[3], 'Maximum angle difference'),
    ]
    if compact:
        parent = ctk.CTkFrame(parent, fg_color="transparent")
        parent.pack(fill="x", pady=(0, 6))
    rows = []
    for title, variable, hint in fields:
        row = ctk.CTkFrame(parent)
        rows.append(row)
        if not compact:
            row.pack(fill='x', padx=12, pady=5)
        row.grid_columnconfigure(0, weight=1)
        WrappedLabel(row, text=title, anchor='w').grid(row=0, column=0, sticky='ew', padx=10, pady=(8, 2))
        ctk.CTkEntry(row, textvariable=variable, width=100).grid(row=0, column=1, padx=10, pady=(8, 2))
        WrappedLabel(row, text=hint, text_color='#B8C0CC', anchor='w', justify='left').grid(
            row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=(2, 8))

    if compact:
        columns_used = [None]
        def arrange(event=None):
            width = event.width if event else 1
            columns = 2 if width / ctk.ScalingTracker.get_widget_scaling(parent) >= 660 else 1
            if columns_used[0] == columns:
                return
            columns_used[0] = columns
            for column in range(2):
                parent.grid_columnconfigure(column, weight=int(column < columns),
                                            uniform="settings" if column < columns else "")
            for index, row in enumerate(rows):
                row.grid(row=index // columns, column=index % columns, sticky="nsew", padx=6, pady=4)
        parent.bind('<Configure>', arrange, add='+')
        arrange()


class ResultPages(ctk.CTkFrame):
    """Show tabs when their measured labels fit; use a menu on narrow windows."""
    def __init__(self, master):
        super().__init__(master, height=100)
        self.pages = {}
        self._navigation_id = None
        self.navigation = ctk.CTkFrame(self, fg_color='transparent')
        self.navigation.pack(fill='x', pady=8)
        self.tab_font = ctk.CTkFont(size=14)
        self.tabs = ctk.CTkSegmentedButton(self.navigation, values=[], command=self.set,
                                          font=self.tab_font, height=34)
        self.selector = ctk.CTkOptionMenu(self.navigation, values=["Summary"], command=self.set, width=200,
                                         font=self.tab_font, height=34)
        self.selector.pack()
        self.navigation.bind('<Configure>', self._arrange_navigation, add='+')
        self.content = ctk.CTkFrame(self, fg_color="transparent", height=1, width=1)
        self.content.pack(fill="both", expand=True)
        self.content.pack_propagate(False)

    def add(self, name):
        page = ctk.CTkFrame(self.content, height=1, width=1)
        self.pages[name] = page
        self.selector.configure(values=list(self.pages))
        self.tabs.configure(values=list(self.pages))
        if len(self.pages) == 1:
            self.set(name)
        else:
            self.tabs.set(self.selector.get())
        self._arrange_navigation()
        # CTk recreates its tab buttons when values change. Measure them once
        # Tk has calculated their requested sizes, including DPI/font scaling.
        if self._navigation_id is not None:
            self.after_cancel(self._navigation_id)
        self._navigation_id = self.after(20, self._refresh_navigation)
        return page

    def _refresh_navigation(self):
        self._navigation_id = None
        self._arrange_navigation()

    def _arrange_navigation(self, event=None):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        width = self.navigation.winfo_width() / scale
        required = self.tabs.winfo_reqwidth() / scale + 24
        show_tabs = bool(self.pages) and width >= required
        visible, hidden = (self.tabs, self.selector) if show_tabs else (self.selector, self.tabs)
        hidden.pack_forget()
        if not visible.winfo_manager():
            visible.pack()

    def set(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        self.selector.set(name)
        self.tabs.set(name)

    def destroy(self):
        if self._navigation_id is not None:
            self.after_cancel(self._navigation_id)
        # CTk 5.2's DropdownMenu.destroy omits its scaling registration cleanup.
        # State switches replace this view; a later DPI change must not address
        # any destroyed native menus. remove_widget is safe if already removed.
        menu = self.selector._dropdown_menu
        ctk.ScalingTracker.remove_widget(menu._set_scaling, menu)
        super().destroy()
