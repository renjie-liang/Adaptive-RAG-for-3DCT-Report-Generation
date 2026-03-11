RET_START_TOKEN = "<|ret_start|>"
RET_END_TOKEN = "<|ret_end|>"
RAG_TOKEN = "[RAG]"


def is_valid_sentence(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if all(c in ' \t\n.,;:!?' for c in s):
        return False
    if s.rstrip(':').lower() in ('lung', 'heart', 'esophagus', 'aorta', 'other', 'findings'):
        return False
    return True
