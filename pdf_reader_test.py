from src.pdf_reader.pdf_parser import PDFParser
from src.pdf_reader.vector_store import VectorStore
import re

# parser = PDFParser()
# store = VectorStore()
# file_text = parser.parse_file('.\documents\WFRP4e-Rulebook_eng.pdf')


# text = file_text[158]
# text = tag_paragraphs(text)
# print(text)

import fitz

doc = fitz.open('.\documents\WFRP4e-Rulebook_eng.pdf')

# human_page = 8
# page = doc.load_page(human_page - 1)

# data = page.get_text("dict")

# for block in data["blocks"]:
#     if "lines" in block:
#         for line in block["lines"]:
#             for span in line["spans"]:
#                 print(span["text"])
#                 bold = bool(int(span["flags"]) & 16)
#                 italic = bool(int(span["flags"]) & 2)
#                 size = span["size"]
#                 print("font:", span["font"])
#                 print("size:", size)
#                 print(f"flags: bold: {bold}, italic: {italic}")
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
