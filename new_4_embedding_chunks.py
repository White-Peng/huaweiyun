import json
import logging
import math
import os
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import xxhash
from FlagEmbedding import FlagAutoModel
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("milvus_bge_pipeline")

# ----------------------------------------------------------------------
# Runtime configuration (override via environment variables)
# ----------------------------------------------------------------------
BGE_CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
BGE_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
BGE_CHUNK_SPECIAL_BUFFER = int(os.getenv("CHUNK_SPECIAL_TOKENS", "2"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "1.92.82.153")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB = os.getenv("MILVUS_DB", "peng")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "peng_conn")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "Full_4_corporas")
# COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "clapnq")
DROP_COLLECTION = os.getenv("DROP_COLLECTION", "true").lower() == "true"

JSONL_PATH = os.getenv(
    "JSONL_PATH",
    os.path.join(os.path.dirname(__file__), "Full_4_corporas.jsonl"),
    # os.path.join(os.path.dirname(__file__), "4_corporas_test_40.jsonl"),
)

BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en")
BGE_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "256"))
INSERT_BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "1000"))
NORMALIZE_EMBED = os.getenv("NORMALIZE_EMBED", "true").lower() == "true"

BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

RUN_SEARCH_TEST = os.getenv("RUN_SEARCH_TEST", "false").lower() == "true"
SEARCH_ALPHA = float(os.getenv("SEARCH_ALPHA", "0.6"))
SEARCH_TOPK = int(os.getenv("SEARCH_TOPK", "5"))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def term_id(term: str) -> int:
    return xxhash.xxh64(term).intdigest() & 0x7FFFFFFF


def bm25_doc_vector(
    tokens: List[str],
    idf: Dict[str, float],
    avgdl: float,
    k: float = BM25_K1,
    b: float = BM25_B,
) -> Dict[int, float]:
    tf = Counter(tokens)
    dl = len(tokens)
    K = k * (1 - b + b * (dl / max(avgdl, 1e-9)))
    vec: Dict[int, float] = {}
    for term, freq in tf.items():
        score = idf.get(term, 0.0) * ((freq * (k + 1)) / (freq + K))
        if score > 0:
            vec[term_id(term)] = float(score)
    return vec


def iter_documents(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("跳过第 %d 行（JSON 解析失败: %s）", line_no, exc)
                continue

            text = (obj.get("text") or obj.get("content") or "").strip()
            if not text:
                continue

            doc_id = (
                obj.get("id")
                or obj.get("_id")
                or obj.get("document_id")
                or f"auto_{line_no}"
            )
            title = (obj.get("title") or "").strip()

            yield {"id": str(doc_id), "title": title, "text": text}


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    effective_max = max(max_tokens - BGE_CHUNK_SPECIAL_BUFFER, 1)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if not token_ids:
        return []

    if len(token_ids) <= max_tokens:
        return [text]

    chunks: List[str] = []
    step = max(max_tokens - overlap_tokens, 1)
    for start in range(0, len(token_ids), step):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(token_ids):
            break
    return chunks


def iter_chunks(
    path: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterator[Dict[str, str]]:
    for doc in iter_documents(path):
        parts = chunk_text(doc["text"], tokenizer, max_tokens, overlap_tokens)
        if not parts:
            continue
        if len(parts) == 1:
            yield {"id": doc["id"], "title": doc["title"], "text": parts[0]}
            continue
        for idx, chunk in enumerate(parts):
            chunk_id = f"{doc['id']}_{idx}"
            title = doc["title"]
            chunk_title = f"{title} [part {idx + 1}]" if title else chunk_id
            yield {"id": chunk_id, "title": chunk_title, "text": chunk}


def scan_corpus_for_stats(
    path: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> Tuple[Dict[str, float], float, int]:
    df = Counter()
    total_len = 0
    doc_count = 0

    for doc in tqdm(
        iter_chunks(path, tokenizer, max_tokens, overlap_tokens),
        desc="Pass 1/2: 统计 BM25 信息",
    ):
        tokens = simple_tokenize(doc["text"])
        doc_count += 1
        total_len += len(tokens)
        if tokens:
            df.update(set(tokens))

    avgdl = total_len / max(doc_count, 1)
    idf = {
        term: math.log((doc_count - df_t + 0.5) / (df_t + 0.5) + 1.0)
        for term, df_t in df.items()
    }
    return idf, avgdl, doc_count


def batched(iterator: Iterable[Dict[str, str]], batch_size: int) -> Iterator[List[Dict[str, str]]]:
    batch: List[Dict[str, str]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_with_bge(
    model: FlagAutoModel,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
    )
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if NORMALIZE_EMBED:
        embeddings = _l2_normalize(embeddings)
    return embeddings.tolist()


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


# ----------------------------------------------------------------------
# Milvus helpers with backward compatibility
# ----------------------------------------------------------------------
def ensure_database() -> None:
    if not hasattr(utility, "list_databases"):
        logger.info("当前 pymilvus 不支持多数据库 API，使用默认数据库")
        return

    base_alias = f"{MILVUS_ALIAS}_base"
    if not connections.has_connection(base_alias):
        connections.connect(alias=base_alias, host=MILVUS_HOST, port=MILVUS_PORT)

    existing = utility.list_databases(using=base_alias)
    if MILVUS_DB not in existing:
        logger.info("创建 Milvus 数据库 %s", MILVUS_DB)
        utility.create_database(MILVUS_DB, using=base_alias)

    connections.disconnect(base_alias)


def connect_milvus() -> None:
    conn_kwargs = {
        "alias": MILVUS_ALIAS,
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
    }
    if hasattr(utility, "list_databases"):
        ensure_database()
        conn_kwargs["db_name"] = MILVUS_DB
    connections.connect(**conn_kwargs)


def ensure_collection(dense_dim: int) -> Collection:
    if utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
        if DROP_COLLECTION:
            logger.info("删除已有集合 %s", COLLECTION_NAME)
            utility.drop_collection(COLLECTION_NAME, using=MILVUS_ALIAS)
        else:
            return Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        ),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(
            name="dense_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dense_dim,
        ),
        FieldSchema(
            name="sparse_embedding",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Hybrid retrieval with BGE dense + BM25 sparse vectors",
    )
    logger.info("创建集合 %s", COLLECTION_NAME)
    return Collection(name=COLLECTION_NAME, schema=schema, using=MILVUS_ALIAS)


def insert_batches(
    collection: Collection,
    model: FlagAutoModel,
    tokenizer,
    idf: Dict[str, float],
    avgdl: float,
    doc_count: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    if doc_count == 0:
        logger.warning("没有可导入的数据，直接退出")
        return

    progress = tqdm(total=doc_count, desc="Pass 2/2: 嵌入 + 插入 Milvus")
    total_inserted = 0

    chunk_iter = iter_chunks(JSONL_PATH, tokenizer, max_tokens, overlap_tokens)
    for batch in batched(chunk_iter, INSERT_BATCH_SIZE):
        texts = [doc["text"] for doc in batch]
        titles = [doc["title"] for doc in batch]
        ids = [doc["id"] for doc in batch]
        tokens_per_doc = [simple_tokenize(text) for text in texts]

        dense_vectors = encode_with_bge(model, texts, BGE_BATCH_SIZE)
        sparse_vectors = [bm25_doc_vector(tokens, idf, avgdl) for tokens in tokens_per_doc]

        collection.insert([ids, titles, texts, dense_vectors, sparse_vectors])
        total_inserted += len(batch)
        progress.update(len(batch))

    progress.close()
    logger.info("已插入 %d 条记录", total_inserted)


def create_indexes(collection: Collection) -> None:
    logger.info("创建 HNSW 索引 (dense)")
    collection.create_index(
        field_name="dense_embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    logger.info("创建倒排索引 (sparse)")
    collection.create_index(
        field_name="sparse_embedding",
        index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
    )
    collection.load()


def hybrid_search(
    collection: Collection,
    model: FlagAutoModel,
    query: str,
    idf: Dict[str, float],
    avgdl: float,
    top_k: int,
    alpha: float,
) -> List[Tuple[str, float, str]]:
    dense_vec = encode_with_bge(model, [query], batch_size=1)[0]
    sparse_vec = bm25_doc_vector(simple_tokenize(query), idf, avgdl)

    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_embedding",
        param={"metric_type": "IP", "params": {}},
        limit=top_k,
    )
    rerank = WeightedRanker(alpha, 1.0 - alpha)
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=top_k,
        output_fields=["title", "text"],
    )

    hits = []
    for hit in results[0]:
        hits.append(
            (
                hit.entity.get("title") or "",
                hit.score,
                (hit.entity.get("text") or "")[:200],
            )
        )
    return hits


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main() -> None:
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"找不到 JSONL 文件：{JSONL_PATH}")

    print(f"CUDA available? {torch.cuda.is_available()}")
    logger.info("加载 BGE 模型：%s", BGE_MODEL_NAME)
    model = FlagAutoModel.from_finetuned(
        BGE_MODEL_NAME,
        use_fp16=torch.cuda.is_available(),
    )
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("无法从 BGE 模型中获取 tokenizer")
    dense_dim_fn = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(dense_dim_fn):
        dense_dim = dense_dim_fn()
    else:
        dense_dim = len(encode_with_bge(model, ["dimension_probe"], 1)[0])
    logger.info("向量维度：%d", dense_dim)

    connect_milvus()
    collection = ensure_collection(dense_dim)

    logger.info("开始扫描语料以计算 BM25 统计")
    idf, avgdl, doc_count = scan_corpus_for_stats(
        JSONL_PATH,
        tokenizer,
        BGE_CHUNK_MAX_TOKENS,
        BGE_CHUNK_OVERLAP,
    )
    logger.info("语料（含切片）数量：%d，平均长度：%.2f", doc_count, avgdl)

    insert_batches(
        collection,
        model,
        tokenizer,
        idf,
        avgdl,
        doc_count,
        BGE_CHUNK_MAX_TOKENS,
        BGE_CHUNK_OVERLAP,
    )
    create_indexes(collection)

    logger.info("数据导入与索引构建完成")

    if RUN_SEARCH_TEST:
        collection.load()
        sample_queries = [
            "What happened after the French Revolution?",
            "How does machine learning work?",
            "Tell me about climate change",
        ]
        for q in sample_queries:
            logger.info("Hybrid search: %s", q)
            for idx, (title, score, snippet) in enumerate(
                hybrid_search(collection, model, q, idf, avgdl, SEARCH_TOPK, SEARCH_ALPHA),
                start=1,
            ):
                print(f"{idx}. [{score:.4f}] {title}\n   {snippet}...\n")
            print("-" * 80)


if __name__ == "__main__":
    main()
