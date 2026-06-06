from smolagents import CodeAgent, HfApiModel,load_tool,tool
from smolagents import LiteLLMModel
import datetime
import requests
import pytz
import yaml
from src.tools.final_answer import FinalAnswerTool
from src.tools.rule_book import RuleBookTool
from src.tools.wfrpsu_itemlist import ItemListTool

from src.Gradio_UI import GradioUI

final_answer = FinalAnswerTool()
rule_book_tool = RuleBookTool()
item_list_tool = ItemListTool()

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompts_ru.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, rule_book_tool, item_list_tool],
    max_steps=10,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()