import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.bge_embedding_function import BGEEmbeddingFunction

sample_texts = [
    "Türkiye Cumhuriyeti devleti bir cumhuriyettir.",
    "Türkiye Devleti, ülkesi ve milletiyle bölünmez bir bütündür.",
    "Her Türk, doğuştan Türkiye Cumhuriyeti vatandaşlığına sahiptir.",
    "İnsan haysiyetiyle bağdaşmayan hiçbir muameleye tabi tutulamaz.",
    "Kanun önünde herkes eşittir.",
]

bge_ef = BGEEmbeddingFunction()

embeddings = bge_ef.model.encode(sample_texts)

print("=" * 60)
print("BGE-m3 Sanity Check")
print("=" * 60)

for i, (text, emb) in enumerate(zip(sample_texts, embeddings["dense_vecs"])):
    print(f"\n[{i+1}] Text: {text}")
    print(f"    Vector shape: {emb.shape}")
    print(f"    First 5 dims: {[round(float(x), 4) for x in emb[:5]]}")
    print(f"    L2 norm: {float((emb ** 2).sum() ** 0.5):.4f}")

print("\n" + "=" * 60)
print("Pairwise cosine similarities:")
print("=" * 60)

import numpy as np

vecs = embeddings["dense_vecs"]
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        cos_sim = float(np.dot(vecs[i], vecs[j]) / (
            (vecs[i] ** 2).sum() ** 0.5 * (vecs[j] ** 2).sum() ** 0.5
        ))
        print(f"  [{i+1}] x [{j+1}]: {cos_sim:.4f}")
