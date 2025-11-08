from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy
from typing import List, Tuple
from tools import *

MODEL_DIR = r"D:\Models\all-MiniLM-L6-v2"
cache_path = r"captions.jsonl"


def extract_keywords(
        text: str,
        kw_model: KeyBERT,
        top_n: int = 5,
        min_ngram: int = 1,
        max_ngram: int = 2
) -> List[Tuple[str, float]]:
    """
    功能：对单条文本用 KeyBERT 抽取关键词短语。
    参数：
        text: 原始 caption 文本
        kw_model: KeyBERT 实例
        top_n: 返回的候选数
        min_ngram/max_ngram: 关键词长度范围（词数）
    返回：
        列表[(关键词, 分数)]
    说明：
        - use_mmr=True 增强多样性；diversity 可按需调节。
    """
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(min_ngram, max_ngram),
        stop_words="english",
        use_maxsum=False,
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )


def pos_filter_terms(terms: List[str], nlp) -> List[str]:
    """
    功能：对候选词做词性过滤，仅保留名词/专有名词/形容词。
    参数：
        terms: 关键词列表（不含分数）
        nlp: spaCy English pipeline
    返回：
        过滤后的关键词列表
    """
    kept = []
    for t in terms:
        doc = nlp(t)
        # 对短语取所有 token 的 POS，只要包含 NOUN/PROPN/ADJ 即保留
        if any(tok.pos_ in {"NOUN", "PROPN", "ADJ"} for tok in doc):
            kept.append(t)
    # 去重并保序
    seen = set()
    uniq = []
    for x in kept:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


if __name__ == "__main__":

    # 下载模型
    # SAVE_DIR = r"D:\Models\all-MiniLM-L6-v2"
    # st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # st.save(SAVE_DIR)

    kw_model = KeyBERT(model=MODEL_DIR)
    nlp = spacy.load("en_core_web_sm")  # spacy英文语言模型，不支持中文处理，中文请移步：spacy-zh-core-web-sm等

    class2caps = load_class_captions(cache_path) #Dict{int: List[str1,str2,str3...]}
    # 每类取所有captions（若缺失就用类名 class1, class2....）
    captions = [
        cap["text"] if isinstance(cap, dict) else cap
        for _, caps in sorted(class2caps.items())
        if isinstance(caps, list)
        for cap in caps
    ]

    for idx, cap in enumerate(captions):
        # 1) KeyBERT 抽取 ngramk控制提取单词与二词短语， top_n表示取前n个
        pairs = extract_keywords(cap, kw_model, top_n=4, min_ngram=1, max_ngram=2)  # [(term, score), ...]
        terms = [w for (w, s) in pairs]

        # 2) POS 过滤（仅保留名词/形容词）
        filtered = pos_filter_terms(terms, nlp)

        # 3) 打印结果
        print(f"[{idx:03d}] caption: {cap}")
        print("    keybert pairs:", pairs)
        print("    keybert terms:", terms) # 未加POS
        print("    filtered:", filtered) # 加了POS过滤
        print("-" * 72)
