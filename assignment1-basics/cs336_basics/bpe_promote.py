from collections import defaultdict, Counter
import regex

def bpe(input_path: str, vocab_max_size: int, special_tokens: list[str]=['|<endoftext>|']) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step 1 初始化词汇表
    vocab_dict: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab_size: int = 256
    
    # dict[tuple, int], tuple: (b'', b'', ...), int: freq
    word_freq = Counter()
    # dict[tuple, int], tuple: (b'', b''), int: freq
    pairs_count = Counter()
    # 已经存在的token
    existing_vocab_set = set(vocab_dict.values())
    # max pair 列表
    # list[bytes, bytes]
    merges: list[bytes, bytes] = []
    pairs_index_word: dict[tuple[bytes], list[tuple[tuple[bytes], int]]] = defaultdict(list)
    
    
    
    # Step 1 加入special token 到词汇表中
    for token in special_tokens:
        if vocab_size >= vocab_max_size:
            break
        if token in existing_vocab_set:
            continue
        token_bytes = token.encode('utf-8')
        existing_vocab_set.add(token_bytes)
        vocab_dict[vocab_size] = token_bytes
        vocab_size += 1
    
    # 加载文本, 过滤文本
    with open(input_path, mode='r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # 根据special 拆开文本为 多个chunks
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)

    # 不同小块进行不同处理
    # Step 2 统计 word freq
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            # 转为 bytes
            word_bytes = word.encode('utf-8')
            word_byte_tuple = tuple(bytes([i]) for i in word_bytes)
            word_freq[word_byte_tuple] += 1
    # Step 2 根据 word freq 统计 pairs freq
    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pairs_count[(word[i], word[i + 1])] += freq
            pairs_index_word[(word[i], word[i + 1])].append((word, i))

    # Step 3 获取 max_pair, 更新vocab dict, merges, 更新 word freq, 更新 pairs freq
    while vocab_size < vocab_max_size:
        # a 获取max_pair
        max_pair = max(pairs_count.items(), key=lambda x: (x[1], x[0]))[0]
        # b 更新merges
        merges.append(max_pair)
        new_token = b''.join(max_pair)
        # c 更新 vocab dict
        vocab_dict[vocab_size] = new_token
        vocab_size += 1
        # d 更新 word freq
        # 判断是否存在
        affect_words: list[tuple[tuple[bytes], int]] = []
        for word, freq in word_freq.items():
            # for i in range(len(word) - 1):
            #     if word[i] + word[i + 1] == new_token:
            has_max_pair = any(word[i] + word[i + 1] == new_token for i in range(len(word) - 1))
            if has_max_pair:
                affect_words.append((word, freq))
        
        # e 更新 word freq的new word和删去old word, 同时 更新pairs count
        for word, freq in affect_words:
            # (1) 更新new word, 删去 old word
            new_word = merge_pair(word, new_token)
            word_freq[new_word] = freq
            del word_freq[word]
            # (2) 更新 pairs count
            # 先删去 word下的所有pair的cnt
            for i in range(len(word) - 1):
                pairs_count[(word[i], word[i + 1])] -= freq
                if pairs_count[(word[i], word[i + 1])] == 0:
                    del pairs_count[(word[i], word[i + 1])]
            # 再加上 new word下的所有pair的cnt
            for i in range(len(new_word) - 1):
                pairs_count[(new_word[i], new_word[i + 1])] += freq
    
    return vocab_dict, merges
                    

def merge_pair(word: tuple[bytes], merged_pair: bytes) -> tuple[bytes]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] + word[i + 1] == merged_pair:
            new_word.append(merged_pair)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


if __name__ == "__main__":
    print('test')
    print(bpe(input_path=r"D:\School\Sustech\AI-Learning\CS336\data\owt_valid.txt",
        vocab_max_size=10000))
    print('end')
    