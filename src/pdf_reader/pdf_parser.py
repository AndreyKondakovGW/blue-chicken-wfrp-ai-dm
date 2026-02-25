from pypdf import PdfReader
import os

class PDFParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        if os.path.exists(file_path):
            reader = PdfReader(file_path)
            return "\n".join([p.extract_text() for p in reader.pages])
        else:
            print(file_path + " Not exist")
            return ""