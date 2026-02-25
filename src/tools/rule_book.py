from smolagents.tools import Tool

RULE_BOOK_DATABASE_PATH = 'wfrp_core_rulebook_eng'

class RuleBookTool(Tool):
    name = "rule_book"
    description = '''Searches the Warhammer Fantasy Roleplay 4th Edition core rulebook for relevant information.\
                    Use this to answer questions about the game rules, character creation, combat, magic, and more.\
                    '''
    inputs = {'query': {'type': 'string', 'description': 'The question or topic to search for in the rulebook.'},
            'rule_book_name': {'type': 'string', 'description': '''The name of the rulebook to search in.\
                            Use wfrp_core_rulebook in most of cases unless it's explicitly stated that you should use alternative rules from Up in Arms.\
                            In that case use wfrp_up_in_arms_rulebook. These rules include changes to how ADVANTAGE functions, how it is gained and lost, how it is spent, and changes to the TALENTS and TRAITS of creatures that interact with ADVANTAGE. '''}}
    output_type = "string"

    def forward(self, query: str, rule_book_name: str) -> str:
        from src.pdf_reader.vecor_store import VectorStore
        vector_store = VectorStore()
        vector_store = vector_store.load_vectorstore(rule_book_name)

        results = vector_store.similarity_search(query, k=2)

        if not results:
            return "No relevant information found in the rulebook."

        response = "Here are some relevant sections from the rulebook:\n\n"
        for res in results:
            response += f"* {res.page_content} [{res.metadata}]\n\n"

        return response.strip()

