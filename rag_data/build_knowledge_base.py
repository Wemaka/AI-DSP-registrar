"""
rag_data/build_knowledge_base.py
Создание RAG-базы знаний из инструкций РЖД.
При отсутствии реального PDF создаёт встроенную базу из текста регламентов.
"""

import re
import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

with open('rag_data/инструкция по сигнализации на жд.txt', 'r', encoding='utf-8') as f:
    EMBEDDED_REGULATIONS = f.read()


def get_regulation_chunks() -> List[dict]:
    """Разберите встроенные правила на фрагменты по разделам."""
    chunks = []
    # Find all sections by their header pattern
    pattern = re.compile(r'=== (РАЗДЕЛ \d+\. [^=]+?) ===', re.MULTILINE)
    matches = list(pattern.finditer(EMBEDDED_REGULATIONS))

    # Add intro chunk
    intro_end = matches[0].start() if matches else len(EMBEDDED_REGULATIONS)
    intro = EMBEDDED_REGULATIONS[:intro_end].strip()
    if intro:
        chunks.append({
            "content": intro,
            "metadata": {"source": "Инструкция РЖД", "section": "Введение", "chunk_id": 0}
        })

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(EMBEDDED_REGULATIONS)
        content = EMBEDDED_REGULATIONS[start:end].strip()

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({
                "content": f"{title}\n{para}",
                "metadata": {
                    "source": "Инструкция по сигнализации РЖД",
                    "section": title,
                    "chunk_id": len(chunks),
                }
            })

    return chunks


def build_faiss_index(chunks: List[dict], embeddings_model=None):
    """Постройте индекс FAIASS из фрагментов, используя трансформаторы предложений."""
    try:
        if embeddings_model is None:
            embeddings_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        texts = [c["content"] for c in chunks]
        vectors = embeddings_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        vectors = np.array(vectors, dtype="float32")

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        index.add(vectors)

        os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(KNOWLEDGE_BASE_PATH, "index.faiss"))
        with open(os.path.join(KNOWLEDGE_BASE_PATH, "chunks.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        return index, chunks, embeddings_model
    except ImportError as e:
        print(f"Зависимость не установлена: {e}. Используется fallback-поиск.")
        return None, chunks, None


def load_or_build(force_rebuild: bool = False):
    index_path = os.path.join(KNOWLEDGE_BASE_PATH, "index.faiss")
    chunks_path = os.path.join(KNOWLEDGE_BASE_PATH, "chunks.pkl")

    chunks = get_regulation_chunks()

    if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            return index, chunks, model
        except Exception:
            pass

    return build_faiss_index(chunks)


def search(query: str, index, chunks: List[dict], model, top_k: int = 3) -> List[dict]:
    """Ищите в базе знаний. Возвращается к поиску по ключевым словам, если FAISS недоступен."""
    if index is not None and model is not None:
        try:
            import numpy as np
            vec = model.encode([query], normalize_embeddings=True)
            vec = np.array(vec, dtype="float32")
            scores, ids = index.search(vec, top_k)
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx >= 0:
                    chunk = chunks[idx].copy()
                    chunk["score"] = float(score)
                    results.append(chunk)
            return results
        except Exception:
            pass

    # Keyword fallback
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk["content"].lower().split())
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


if __name__ == "__main__":
    print("Строим базу знаний...")
    idx, ch, mdl = build_faiss_index(get_regulation_chunks())
    print(f"Добавлено чанков: {len(ch)}")
    if idx is not None:
        results = search("что делать при заклинивании колёсной пары", idx, ch, mdl)
        print(f"\nТест поиска (топ-{len(results)}):")
        for r in results:
            print(f"  [{r['metadata']['section']}] score={r.get('score', 'n/a'):.3f}")
            print(f"  {r['content'][:100]}...")

