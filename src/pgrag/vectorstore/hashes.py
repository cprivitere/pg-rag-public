import hashlib
import json


def embedding_hash(document):
    content = json.dumps(
        {"id": document["id"], "text": document["text"]}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def metadata_hash(document):
    content = json.dumps(document.get("metadata", {}), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
