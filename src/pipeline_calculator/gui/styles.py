"""Reusable ttk styles, scoped to the interpreter and rendered pixel sizes."""
from tkinter import ttk


def dark_style(widget):
    style = ttk.Style(widget)
    if style.theme_use() != 'clam':
        style.theme_use('clam')
    return style


def table_styles(widget, factor, row_height, *, compact=False):
    style = dark_style(widget)
    pixels = (round(13 * factor), round(row_height * factor), round(14 * factor),
              round((12 if compact else 8) * factor), round(5 * factor))
    name = 'Pipeline' + '_'.join(map(str, pixels)) + ('Compact' if compact else '')
    root = widget._root()
    configured = root.__dict__.setdefault('_pipeline_table_styles', set())
    if name not in configured:
        tree = name + '.Treeview'
        style.configure(tree, background='#242424', foreground='#EEEEEE', fieldbackground='#242424',
                        borderwidth=0, lightcolor='#242424', darkcolor='#242424',
                        rowheight=pixels[1], font=('Arial', -pixels[0]))
        style.configure(tree + '.Heading', background='#343434', foreground='#EEEEEE', relief='solid',
                        borderwidth=1, bordercolor='#535B65', lightcolor='#535B65', darkcolor='#535B65',
                        font=('Arial', -pixels[0], 'bold'), padding=(pixels[3], pixels[4]))
        if compact:
            # Data cells have a built-in text inset. Account for it so their
            # text aligns with the heading's border and padding.
            style.configure(tree + '.Cell', padding=(max(0, pixels[3] - 1), pixels[4]),
                            font=('Arial', -pixels[0]))
            style.layout(tree + '.Cell', [
                ('Treeitem.padding', {'sticky': 'nswe', 'children': [
                    ('Treeitem.text', {'sticky': 'we'}),
                ]}),
            ])
        style.layout(tree + '.Heading', [
            ('Treeheading.cell', {'sticky': 'nswe'}),
            ('Treeheading.border', {'sticky': 'nswe', 'children': [
                ('Treeheading.padding', {'sticky': 'nswe', 'children': [
                    ('Treeheading.image', {'side': 'right', 'sticky': 'e'}),
                    ('Treeheading.text', {'sticky': 'we'}),
                ]}),
            ]}),
        ])
        style.map(tree, background=[('selected', '#1F538D')],
                  foreground=[('disabled', '#B6C0CE'), ('selected', '#FFFFFF')])
        style.map(tree + '.Heading', background=[('pressed', '#253C50'), ('active', '#404040')],
                  foreground=[('disabled', '#B6C0CE'), ('active', '#FFFFFF')])
        for direction in ('Vertical', 'Horizontal'):
            scrollbar = name + '.' + direction + '.TScrollbar'
            style.configure(scrollbar, background='#454545', troughcolor='#242424', arrowcolor='#EEEEEE',
                            bordercolor='#242424', lightcolor='#454545', darkcolor='#454545', arrowsize=pixels[2])
            style.map(scrollbar, background=[('pressed', '#637083'), ('active', '#535F6E')],
                      arrowcolor=[('disabled', '#B6C0CE')])
        configured.add(name)
    return name


def corridor_button_style(widget, factor):
    style = dark_style(widget)
    name = f'Corridor{round(13 * factor)}.TButton'
    style.configure(name, background='#206CA4', foreground='#FFFFFF', borderwidth=0,
                    padding=(8, 2), anchor='center', font=('Arial', -round(13 * factor)))
    style.map(name, background=[('disabled', '#343D47'), ('pressed', '#174D76'), ('active', '#185888')],
              foreground=[('disabled', '#B6C0CE'), ('pressed', '#FFFFFF'), ('active', '#FFFFFF')])
    return name


def scope_style(widget, factor):
    style = dark_style(widget)
    name = f'StateScope{round(14 * factor)}_{round(5 * factor)}.TCombobox'
    style.configure(name, foreground='#F1F4F8', fieldbackground='#18364D', background='#206CA4',
                    arrowcolor='#FFFFFF', bordercolor='#4696CF', lightcolor='#4696CF',
                    darkcolor='#4696CF', padding=round(5 * factor),
                    arrowsize=round(14 * factor))
    style.map(name, fieldbackground=[('disabled', '#343D47'), ('readonly', '#18364D')],
              foreground=[('disabled', '#B6C0CE'), ('readonly', '#F1F4F8')],
              selectforeground=[('readonly', '#FFFFFF')], selectbackground=[('readonly', '#18364D')],
              bordercolor=[('focus', '#A9D9FF')],
              background=[('disabled', '#343D47'), ('pressed', '#174D76'),
                          ('active', '#2F83BD'), ('readonly', '#206CA4')])
    return name
