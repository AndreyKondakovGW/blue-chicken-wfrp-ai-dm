import requests
from bs4 import BeautifulSoup

from rapidfuzz import process, fuzz
import re
import pandas as pd
from src.tools.wfrpsu_itemlist import ItemListTool

il = ItemListTool()
talants = """Гномий двуручный молот"""
for t in talants.split(','):
    info = il.forward(item_name=t, item_list_name="item-melee-weapon")
    print(info)