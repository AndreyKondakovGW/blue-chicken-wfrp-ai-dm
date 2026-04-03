from src.pdf_reader.pdf_parser import PDFParser
from src.pdf_reader.vector_store import VectorStore
import re

import fitz

doc = fitz.open('.\documents\WFRP4e-Rulebook_eng.pdf')

import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
chars_per_size = defaultdict(int)

for page in doc:
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                size = round(span["size"], 1)
                text = span["text"]
                
                chars_per_size[size] += len(text)

sizes = sorted(chars_per_size.keys())
counts = [chars_per_size[s] for s in sizes]

plt.bar(sizes, counts)
plt.xlabel("Font size")
plt.xticks(sizes)  # Set x-ticks to be the font sizes
plt.ylabel("Total characters")
plt.yscale("log")  # Use logarithmic scale for better visibility
plt.title("Total Characters per Font Size")
plt.show()
