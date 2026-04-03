import requests
from bs4 import BeautifulSoup

from rapidfuzz import process, fuzz
import re
import pandas as pd
from src.tools.wfrpsu_itemlist import ItemListTool

il = ItemListTool()
talants = """быстрая реакция, грамотность, меткость, непреклонность, прирождённый воин, рокировка, фортуна, 
чтение по губам, шестое чувство"""
for t in talants.split(','):
    info = il.forward(item_name=t, item_list_name="talent")
    print(info)