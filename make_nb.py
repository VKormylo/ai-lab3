"""Конвертує lab3.py (клітинки '# %%' / '# %% [markdown]') у ноутбук lab3.ipynb.

Запуск: python make_nb.py
"""
import re
import nbformat

src = open("lab3.py", encoding="utf-8").read()
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for header, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        text = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    elif body.strip():
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, "lab3.ipynb")
print("клітинок:", len(cells))
