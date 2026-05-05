import re
import math
from collections import Counter
from typing import List, Optional


class Compressor:
    """
    极致上下文压缩器
    核心策略：TF-IDF 句子评分 + 位置加权，选出最重要的句子保留。
    """

    STOP_WORDS = {
        # English
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'out',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up', 'it',
        'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
        'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
        'them', 'their', 'what', 'which', 'who', 'whom',
        # Chinese
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
        '一', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '什么', '那',
        '啊', '呢', '吧', '吗', '哦', '嗯', '呀', '哈', '嘛', '啦',
        '哪', '怎么',
    }

    def __init__(self):
        pass

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        en_words = re.findall(r'[a-zA-Z]+', text.lower())
        tokens.extend([w for w in en_words if w not in self.STOP_WORDS and len(w) > 1])
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        for i in range(len(cn_chars)):
            tokens.append(cn_chars[i])
            if i < len(cn_chars) - 1:
                tokens.append(cn_chars[i] + cn_chars[i + 1])
        return tokens

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def score_sentences(self, sentences: List[str]) -> List[float]:
        if not sentences:
            return []

        all_tokens = []
        sent_tokens_list = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            sent_tokens_list.append(tokens)
            all_tokens.extend(tokens)

        word_freq = Counter(all_tokens)
        total = len(all_tokens) if all_tokens else 1
        n = len(sentences)

        scores = []
        for i, tokens in enumerate(sent_tokens_list):
            if not tokens:
                scores.append(0.0)
                continue
            score = sum(
                (word_freq[t] / total) * math.log(total / word_freq[t])
                for t in tokens
            )
            # 位置加权：首尾句更重要
            pos_w = 1.2 if i == 0 else (1.1 if i == n - 1 else 1.0)
            # 长度因子：太短的句子权重低
            len_w = min(len(tokens) / 5.0, 1.0)
            scores.append(score * pos_w * len_w)
        return scores

    def compress(self, text: str, ratio: float = 0.3,
                 min_sentences: int = 2, max_tokens: int = 200) -> str:
        """
        压缩文本，保留 ratio 比例的最重要句子。

        Args:
            text: 原始文本
            ratio: 保留比例 (0.1-0.9)
            min_sentences: 最少保留句数
            max_tokens: 最大 token 数限制
        """
        if not text or not text.strip():
            return text

        sentences = self.split_sentences(text)
        if not sentences:
            return text

        ratio = max(0.1, min(0.9, ratio))
        target = max(min_sentences, int(len(sentences) * ratio))

        scores = self.score_sentences(sentences)
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = sorted(idx for idx, _ in scored[:target])

        result = ' '.join(sentences[i] for i in selected)

        # 检查 token 限制
        tokens = self.estimate_tokens(result)
        if tokens > max_tokens:
            while len(selected) > 1 and self.estimate_tokens(result) > max_tokens:
                # 移除得分最低的
                worst = min(selected, key=lambda i: scores[i])
                selected.remove(worst)
                result = ' '.join(sentences[i] for i in selected)

        return result

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        cn = len(re.findall(r'[\u4e00-\u9fff]', text))
        en = len(re.findall(r'[a-zA-Z]+', text))
        num = len(re.findall(r'[0-9]', text))
        return cn + en + num

    @staticmethod
    def confusion_score(text: str) -> float:
        """
        检测文本是否像 LLM 的困惑/澄清回复。
        返回 0.0-1.0，越高越可能是困惑回复。
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0

        # 英文困惑模式
        en_patterns = [
            (r'could you (clarify|elaborate|explain|provide)', 0.8),
            (r'what (do you mean|are you asking|specifically)', 0.7),
            (r"i('|\s)m not (sure|certain|clear) (what|how|why)", 0.7),
            (r'can you (be more specific|clarify|elaborate)', 0.8),
            (r"i don't (understand|follow|get)", 0.6),
            (r'could (you|you please) (rephrase|clarify)', 0.8),
            (r'(what|which) (exactly|specifically|precisely)', 0.5),
            (r'(unclear|ambiguous|vague|confusing)', 0.6),
            (r'need more (context|information|details)', 0.8),
            (r"(sorry|apologies),?\s*(but )?(I |i )?(can|could|don't)", 0.4),
            (r'(repeat|restate|rephrase)', 0.6),
        ]

        # 中文困惑模式
        cn_patterns = [
            (r'你(能|能否)(具体|详细|进一步)(说明|解释|描述)', 0.9),
            (r'不太(明白|理解|清楚)你(的|说的|想)', 0.8),
            (r'(请问|麻烦)(你 |您 )?(能|能否)(再说|重新|具体)', 0.8),
            (r'(可以|能否)(更(详细|具体|清楚)|重新)(说明|解释|描述)', 0.9),
            (r'不太(懂|明白)', 0.7),
            (r'什么(意思|情况|问题)', 0.5),
            (r'(没有|缺少)(足够|更多|上下文|背景)(信息|内容|资料)', 0.9),
            (r'(抱歉|不好意思).{0,10}(不太|没)(明白|理解|懂)', 0.8),
            (r'(能|能否)(再(说|解释|描述)(一下|清楚))', 0.6),
        ]

        for pattern, weight in en_patterns + cn_patterns:
            if re.search(pattern, text_lower):
                score = max(score, weight)

        # 短回复通常是困惑（LLM 正常回复通常较长）
        if len(text.strip()) < 20:
            score = max(score, 0.3)

        return min(score, 1.0)
