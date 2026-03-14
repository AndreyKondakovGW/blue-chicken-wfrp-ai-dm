from pypdf import PdfReader
import os

import fitz

class PDFParser:
    def __init__(self):
        pass

    # def parse_file(self, file_path):
    #     if os.path.exists(file_path):
    #         reader = PdfReader(file_path)
    #         return {i+1: p.extract_text() for i, p in enumerate(reader.pages)}
    #     else:
    #         print(file_path + " Not exist")
    #         return ""

    def parse_file(self, pdf_path, tol=0.5):
        doc = fitz.open(pdf_path)
        pages_dict = {}

        def different(a, b):
            return abs(a - b) > tol

        for page_num, page in enumerate(doc):
            lines = []

            blocks = page.get_text("dict")["blocks"]

            # collect lines
            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    text = ""
                    sizes = []

                    for span in line["spans"]:
                        text += span["text"]
                        sizes.append(span["size"])

                    if text.strip():
                        avg_size = sum(sizes) / len(sizes)
                        lines.append((text.strip(), avg_size))

            # detect special lines
            marked_lines = []

            for i in range(len(lines)):
                text, size = lines[i]

                prev_size = lines[i-1][1] if i > 0 else None
                next_size = lines[i+1][1] if i < len(lines)-1 else None

                if prev_size and next_size:
                    if different(size, prev_size) and different(size, next_size):
                        marked_lines.append("<p> " + text)
                    else:
                        marked_lines.append(text)
                else:
                    marked_lines.append(text)

            pages_dict[page_num+1] = "\n".join(marked_lines)

        return pages_dict