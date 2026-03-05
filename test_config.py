from src.providers import get_llm_model
from src.settings import load_settings

try:
    settings = load_settings()
    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {settings.llm_model}")
    model = get_llm_model()
    print(f"Successfully initialized {type(model).__name__}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
