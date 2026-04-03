from smolagents.tools import Tool
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
import re
import os
import pandas as pd
import requests

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def find_item_id(query, items, threshold=75):
    names = list(items.keys())
    matches = process.extract(
        query,
        names,
        scorer=fuzz.WRatio,
        limit=5
    )
    results = [
        (name, items[name], score)
        for name, score, _ in matches
        if score >= threshold
    ]
    return results

def load_content_table(url, output_name):
    dfs = []
    page = 1
    last_id = -1
    while True:
        url = url + f"&page={page}"
        try:
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table")

            rows = []
            for tr in table.find_all("tr"):
                if tr and tr.has_attr("data-key"):
                    data_key = tr["data-key"]
                    cells = [data_key] + [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                else:
                    cells = [0] + [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(cells)
            if last_id == rows[-1][0]:
                break
            else:
                last_id = rows[-1][0]
            df = pd.DataFrame(rows[1:], columns=rows[0])

            if df is None or df.empty:
                break

            dfs.append(df)
            page += 1

        except Exception as e:
            print(f"Stopping because of error: {e}")
            break

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(output_name, index=False)
    return final_df

def load_item_info(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    item_descrition = ""
    for tr in table.find_all("tr"):
        row = []
        for td in tr.find_all(["td", "th"]):
            span = td.find("span")
            
            if span and span.has_attr("data-bs-content"):
                name = span.get_text(strip=True)
                info = span["data-bs-content"]
                value = f"{name}: {info}"
            else:
                value = td.get_text(strip=True)
            
            row.append(value)
    
        item_descrition += f"{row[0]}: {row[1]} \n"
    return item_descrition

class ItemListTool(Tool):
    name = "item_list"
    description = '''Searches insidd Item and Ability lists form wfrp.su website.\
                    Use this to answer questions about the weapon stats traits and cost amd descriptions for spells and talants \
                    !Items and abilites description will be written using Russian language!
                    '''
    inputs = {'item_name': {'type': 'string', 'description': 'The name you want to ask information about.'},
            'item_list_name': {'type': 'string', 'description': '''The name of item list you can search information in. Here is avaliable lists:\
                            witem-melee-weapon for melee weapon, item-ranged-weapon for ranged weapon, talent for character talents, spells for magic spells, miracle for priests miracles'''}}
    output_type = "string"

    def forward(self, item_name: str, item_list_name: str) -> str:
        import requests
        import os
        import pandas as pd

        if re.search(r'[А-Яа-яЁё]', item_name):
            lang = "rus"
        elif re.search(r'[A-Za-z]', item_name):
            lang = "eng"
        else:
            print("No Russian or English letters")

        if os.path.exists(f"./databases/{item_list_name}_{lang}.csv"):
            df = pd.read_csv(f"./databases/{item_list_name}_{lang}.csv")
            if lang == "rus":
                name2id = df.set_index("Название")["0"].to_dict()
            else:
                name2id = df.set_index("Название (англ.)")["0"].to_dict()
        else:
            url = f"https://wfrp.su/{item_list_name}/index?hide_home_rules=1&per-page=100"
            #html = requests.get(url).text
            df = load_content_table(url, f"./databases/{item_list_name}_{lang}.csv")
            if lang == "rus":
                name2id = df.set_index("Название")[0].to_dict()
            else:
                name2id = df.set_index("Название (англ.)")[0].to_dict()
        
        items_candidates = find_item_id(normalize(item_name), name2id)
        if len(items_candidates) >0:
            item_info = "Closest items descriptions: \n"
            for item in items_candidates:
                url = f"https://wfrp.su/{item_list_name}/view/{item[1]}"
                html = requests.get(url).text
                item_info += load_item_info(html)
            return item_info
        else:
            return "Unfortunately, no items with this names of with similar names are found in the base"