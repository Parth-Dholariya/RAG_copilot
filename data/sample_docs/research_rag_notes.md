# Retrieval Augmented Generation Notes

Retrieval augmented generation improves factuality by grounding model responses in external documents. A typical RAG pipeline loads documents, splits them into chunks, embeds each chunk, stores the embeddings in a vector database, and retrieves relevant context at query time.

Semantic chunking preserves coherent sections better than fixed-size splitting. Chunk overlap helps avoid losing important facts at boundaries, but too much overlap can increase index size and duplicate retrieval results.

Evaluation should measure retrieval quality and answer quality. Recall@K checks whether a relevant document appears in the top retrieved contexts. Mean reciprocal rank rewards systems that place the first relevant result near the top. Faithfulness measures whether answer claims are supported by retrieved evidence.

Citation-grounded generation reduces hallucination risk because users can inspect the source context behind the answer.
