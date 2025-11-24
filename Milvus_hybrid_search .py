import json
import logging
import math
import os
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import xxhash
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("milvus_ingest")

# ----------------------- Runtime configuration -----------------------
MILVUS_HOST = os.getenv("MILVUS_HOST", "1.92.82.153")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB = os.getenv("MILVUS_DB", "peng")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "peng_conn")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "clapnq")
DROP_COLLECTION = os.getenv("DROP_COLLECTION", "true").lower() == "true"

JSONL_PATH = os.getenv(
    "JSONL_PATH",
    os.path.join(os.path.dirname(__file__), "4_corporas_test_40.jsonl"),
)

BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
BGE_DEVICE = os.getenv("BGE_DEVICE")
BGE_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "128"))
INSERT_BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "512"))

BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))


# ----------------------------- Helpers ------------------------------
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


def iter_jsonl(path: str) -> Iterator[Dict[str, str]]:
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


def scan_corpus_for_stats(path: str) -> Tuple[Dict[str, float], float, int]:
    df = Counter()
    total_len = 0
    doc_count = 0

    for doc in tqdm(iter_jsonl(path), desc="Pass 1/2: 统计 BM25 信息"):
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
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings.tolist()


def ensure_database() -> None:
    base_alias = f"{MILVUS_ALIAS}_base"
    if not connections.has_connection(base_alias):
        connections.connect(alias=base_alias, host=MILVUS_HOST, port=MILVUS_PORT)
    existing = utility.list_databases(using=base_alias)
    if MILVUS_DB not in existing:
        logger.info("创建 Milvus 数据库 %s", MILVUS_DB)
        utility.create_database(MILVUS_DB, using=base_alias)
    connections.disconnect(base_alias)


def connect_milvus() -> None:
    ensure_database()
    if connections.has_connection(MILVUS_ALIAS):
        connections.disconnect(MILVUS_ALIAS)
    connections.connect(
        alias=MILVUS_ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        db_name=MILVUS_DB,
    )


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
    model: SentenceTransformer,
    idf: Dict[str, float],
    avgdl: float,
    doc_count: int,
) -> None:
    if doc_count == 0:
        logger.warning("没有可导入的数据，直接退出")
        return

    progress = tqdm(
        total=doc_count,
        desc="Pass 2/2: 嵌入 + 插入 Milvus",
    )
    total_inserted = 0

    for batch in batched(iter_jsonl(JSONL_PATH), INSERT_BATCH_SIZE):
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


def main() -> None:
    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"找不到 JSONL 文件：{JSONL_PATH}")

    logger.info("加载 BGE 模型：%s", BGE_MODEL_NAME)
    model = SentenceTransformer(BGE_MODEL_NAME, device=BGE_DEVICE)
    dense_dim = model.get_sentence_embedding_dimension()
    logger.info("向量维度：%d", dense_dim)

    connect_milvus()
    collection = ensure_collection(dense_dim)

    logger.info("开始扫描语料以计算 BM25 统计")
    idf, avgdl, doc_count = scan_corpus_for_stats(JSONL_PATH)
    logger.info("语料文档数：%d，平均长度：%.2f", doc_count, avgdl)

    insert_batches(collection, model, idf, avgdl, doc_count)
    create_indexes(collection)

    logger.info("数据导入与索引构建完成")


if __name__ == "__main__":
    main()
