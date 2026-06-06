from smolagents.tools import Tool
from src.pdf_reader.hybrid_rag_search import *
RULE_BOOK_DATABASE_PATH = 'wfrp_core_rulebook_eng'

class RuleBookTool(Tool):
    name = "rule_book"
    description = '''Ищет релевантную информацию в основной книге правил Warhammer Fantasy Roleplay 4th Edition.
    Используй этот инструмент для ответов на вопросы о правилах игры, создании персонажей, боевой системе, магии и других игровых механиках.
    В ответе ты всегда получишь номер страницы, из которой взята информация. Эта информация крайне важна — используй её в ответе каждый раз. Для поиска используй запросы на русском языке.
                    '''
    inputs = {'query': {'type': 'string', 'description': 'Вопрос или тема которую нужно найти в книге правил. Данный вопрос должен быть на РУССКОМ ЯЗЫКЕ'},
            'rule_book_name': {'type': 'string', 'description': '''Имя книги правил, в которой выполняется поиск.
            В большинстве случаев используй WFRPG4E_ru, если явно не указано иное.
            Если указано использование альтернативных правил из Up in Arms, используй up_in_arms_ru.
            Эти правила включают изменения механики ADVANTAGE: как он получается, теряется и расходуется, а также изменения ТАЛАНТОВ и ЧЕРТ существ, взаимодействующих с ADVANTAGE.
            ''' }}
    output_type = "string"
    #Важно: текст правил на русском языке.
            #Если используешь wfrp_core_rulebook_ru — отвечай на русском языке.
            #Если используешь wfrp_core_rulebook (английскую версию) — отвечай на английском языке.


    def forward(self, query: str, rule_book_name: str) -> str:
        from src.pdf_reader.vector_store import VectorStore

        rerancker = Rerancker()
        embedder = OllamaEmbeddings(model="bge-m3")
        retriver = HybridRetrival(dense_embedder=embedder)
        retriver.load_vectorstore(rule_book_name)


        #vector_store = VectorStore()
        #vector_store = vector_store.load_vectorstore(rule_book_name)
        
        results = retriver.search(query, k=3)
        #results = vector_store.similarity_search_with_score(query.lower(), k=6)

        # for doc, score in results:
        #     print("Score:", score)
        #     print("Content:", doc.page_content[:30])
        #     print("---")
        # results = vector_store.similarity_search(query, k=6)
        if not results:
            return "No relevant information found in the rulebook."

        response = "Here are some relevant sections from the rulebook:\n\n"
        
        for res in results:
            response += f"* {res}\n\n"

        return response.strip()

