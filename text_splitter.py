from typing import List

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of `chunk_size` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks
