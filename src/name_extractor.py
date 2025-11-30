from gliner2 import GLiNER2

def extract_name_gliner(text: str) -> str:
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    result = extractor.extract_entities(text[:700], ["person"])
    return result["entities"]