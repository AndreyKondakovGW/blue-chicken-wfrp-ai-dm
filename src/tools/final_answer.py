from typing import Any, Optional
from smolagents.tools import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Предоставляет финальный ответ на поставленную задачу."
    inputs = {'answer': {'type': 'any', 'description': '''Должен формировать полный и исчерпывающий ответ. Если использовались источники, обязательно укажи это в ответе.'''}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
