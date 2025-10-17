from collections import defaultdict, Counter
from tqdm import tqdm
import regex

class BPE:
    def __init__(self, input_path=None, vocab_max_size=257, 
                 special_tokens: list[str]=['|<endoftext>|']):
        self.vocab_dict: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256
        self.vocab_max_size = vocab_max_size
        self.special_tokens = special_tokens
        # 预处理后的tokenization
        self.pre_tokenized = Counter()
        # Pair 计数器
        self.pairs_counter = Counter()
        # map str: tuple, "hello": (h,e,ll,o)
        self.map_word = dict()
        # 合并对
        self.merges_pair: list[tuple[bytes, bytes]] = []
        # 目前最大的pair, 以及对应的出现的次数
        self.max_pair, self.max_value = None, 0
        # 获取text
        self.text = None
        self.input_path = input_path

        # print('--------------init-------------------------')
        self._init_text()
        self._init_pre_tokenized()
        self._init_pairs_counter()
        self._init_vocab_dict()
        
        # print('-------------end init---------------------------')
        
    def _init_pre_tokenized(self):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        chunks = regex.split('|'.join(map(regex.escape, self.special_tokens)), self.text) #首先按照特殊字符进行大分割，比如<endoftext>按照章节分割

        
        # for s in tqdm(re.finditer(PAT, self.text), desc="re.finditer", unit='word'):
        for chunk in chunks:
            for word in regex.finditer(PAT, chunk):
                # word_bytes = s.group().encode('utf-8')
                # key = tuple(b for b in word_bytes)
                # self.pre_tokenized[key] += 1
                # self.map_word[word_bytes] = key    
                key = tuple(bytes([c]) for c in word.group().encode('utf-8'))
                self.pre_tokenized[key] += 1
                self.map_word[word.group().encode('utf-8')] = key # 初次匹配
        
        # print("now pre tokenization", self.pre_tokenized)
        # print("map word", self.map_word)
        
    # TODO: 后续看下是否要把这个tqdm去掉
    def _init_pairs_counter(self):
        # for k, v in tqdm(self.pre_tokenized.items(), desc="Counting pairs...", unit='words', ncols=80):
        for k, v in self.pre_tokenized.items():
            self.update_counter(k, v)
        # print("pairs counter", self.pairs_counter)
        
    def _init_vocab_dict(self):
        if self.vocab_size >= self.vocab_max_size:
            return
        existing_byte_values: set[bytes] = set(self.vocab_dict.values())
        for s in self.special_tokens:
            if self.vocab_size >= self.vocab_max_size:
                break
            s_bytes = s.encode('utf-8')
            if s_bytes not in existing_byte_values:
                self.vocab_dict[self.vocab_size] = s_bytes
                existing_byte_values.add(s_bytes)
                self.vocab_size += 1
            
    
    def _init_text(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        # print(f"The num char is: {len(self.text):,}")
        # print('read success')

    
    def update_counter(self, word, num, merge_pair=None):
        # merge_pair: 合并后的pair
        # pair_1: 合并的第一个str
        # pair_2: 合并的第二个str
        # 只更新一个word
        if merge_pair:
            # print(merge_pair)
            pair_1, pair_2 = self.max_pair
            # 先减去原有的 pair_1, pair_2
            for s in word:
                if s == merge_pair:
                    self.pairs_counter[(pair_1, pair_2)] -= num
            if self.pairs_counter[(pair_1, pair_2)] == 0:
                del self.pairs_counter[(pair_1, pair_2)]
    
        for c1, c2 in zip(word[:-1], word[1:]):
            if not merge_pair:
                self.pairs_counter[(c1, c2)] += num
            elif c1 == merge_pair and c2 != merge_pair:
                self.pairs_counter[(pair_2, c2)] -= num
                if self.pairs_counter[(pair_2, c2)] == 0:
                    del self.pairs_counter[(pair_2, c2)]
                self.pairs_counter[(merge_pair, c2)] += num
            elif c2 == merge_pair and c1 != merge_pair:
                self.pairs_counter[(c1, pair_1)] -= num
                if self.pairs_counter[(c1, pair_1)] == 0:
                    del self.pairs_counter[(c1, pair_1)]
                self.pairs_counter[(c1, merge_pair)] += num
            elif c1 == merge_pair and c2 == merge_pair:
                self.pairs_counter[(c1, c2)] += num
                            
                
    def _get_max_pair(self):
        self.max_pair, self.max_value = max(
            self.pairs_counter.items(),
            key=lambda x: (x[1], x[0])  # 次数优先，pair 本身做 tie-breaker
        )
        # self.max_pair, self.max_value = max(reversed(self.pairs_counter.items()), key=lambda x: x[1])
        # print('now max pair', self.max_pair)
    
    def _update_vocab_dict(self):
        # print(self.max_pair)
        self.vocab_dict[self.vocab_size] = b''.join(self.max_pair)
        self.vocab_size += 1
    
    def _get_words_with_max_pair(self):
        match_str = b"".join(self.max_pair)
        for key, word in self.map_word.items():
            if match_str in key:
                yield key, word
    
    def _update_pre_token(self, old_word: tuple):
        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and (old_word[i], old_word[i + 1]) == self.max_pair:
                new_word.append(b"".join(self.max_pair))
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        new_word = tuple(new_word)
        self.pre_tokenized[new_word] = self.pre_tokenized[old_word]
        del self.pre_tokenized[old_word]
        # print(f'update pre token old word:{old_word} -> new_word:{new_word}')
        return new_word
    
    def _update_pairs_counter(self, new_word: tuple):
        self.update_counter(new_word, self.pre_tokenized[new_word], 
                            merge_pair=b''.join(self.max_pair),
                            )
    
    def _update_map_word(self, key: str, new_word: tuple):
        self.map_word[key] = new_word
    
    def __call__(self):
        while self.vocab_size < self.vocab_max_size:
        # for _ in tqdm(range(self.vocab_max_size - self.vocab_size), desc="Train BPE", ncols=80,
        #               unit='words'):
            # print(f"------------------{self.vocab_size}-----------------")
            # 1. get max pair
            self._get_max_pair()
            # 更新vocab 字典
            self._update_vocab_dict()
            # 更新merge_pair
            self.merges_pair.append(self.max_pair)
            # 2. get all words with max_pair, 
            #    for word in words:  update pre_token, update pairs_counter, update_map_words
            for key, word in self._get_words_with_max_pair():
                new_word = self._update_pre_token(word)
                self._update_pairs_counter(new_word)
                self._update_map_word(key, new_word)
            # print('now counter :', self.pairs_counter)
            # print('now map:', self.map_word)
            # print('-----------------------------------------------------')
        return self.vocab_dict, self.merges_pair

if __name__ == "__main__":
    # print('program start run.')
    bpe = BPE(vocab_max_size=30000, input_path=r"D:\School\Sustech\AI-Learning\CS336\data\TinyStoriesV2-GPT4-valid.txt")
    bpe()
    # print('program end run.')

