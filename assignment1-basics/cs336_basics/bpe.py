from collections import defaultdict, Counter
import regex
from typing import BinaryIO
import os
from tqdm import tqdm
import time
import json
import pickle
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, ByteString, Union


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes = b"<|endoftext|>",
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# --- 定义任务函数 ---
def process_chunk(chunk: str) -> Dict[Tuple[ByteString, ...], int]:
    """
    处理单个文本块，进行预分词、编码和局部词频统计。
    """
    local_word_freq: Dict[Tuple[ByteString, ...], int] = defaultdict(int)
    
    # 小分割，按照空格和标点
    for word in regex.finditer(PAT, chunk):
        # 对每一个单词进行编码，并转换为bytes
        word_bytes = word.group().encode("utf-8")
        
        # 将字节流转换为元组，e.g. ('h', 'e', 'l', 'l', 'o')
        # 注意: 这里的元素是bytes类型，如 b'h'
        bytes_list_tuple = tuple(bytes([x]) for x in word_bytes)
        
        # 统计每个token出现的频率
        local_word_freq[bytes_list_tuple] += 1
        
    return local_word_freq

# --- 主执行逻辑 ---
def parallel_tokenize_and_count(input_path: str, special_tokens: List[str], PAT: str) -> Dict[Tuple[ByteString, ...], int]:
    """
    读取文件，切分文本，并使用多进程进行并行分词和词频统计。
    """
    # 第1步：读取整个文件内容 (保持单线程，通常I/O不是瓶颈)
    print(f"Reading file: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return defaultdict(int)
    
    # 第2步：对语料库里的文段进行大分割
    # 首先按照特殊字符进行大分割
    chunks: List[str] = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    
    # 过滤掉空的chunk
    chunks = [c for c in chunks if c.strip()]
    
    # 确定使用的进程数，通常是CPU核心数
    num_processes: int = os.cpu_count() or 4


    final_word_freq: Dict[Tuple[ByteString, ...], int] = defaultdict(int)
    
    # 第3步：使用进程池并行处理 chunks
    with Pool(processes=num_processes) as pool:
        # map函数将process_chunk应用到chunks列表中的每一个元素上
        # 结果是一个包含每个进程返回的局部词频字典的列表
        list_of_local_freqs: List[Dict[Tuple[ByteString, ...], int]] = pool.map(process_chunk, chunks)
        
    # 第4步：合并所有局部词频统计结果
    for local_freq in list_of_local_freqs:
        for token, count in local_freq.items():
            final_word_freq[token] += count
            
    return final_word_freq


def process_file_chunk(file_path: str, start_offset: int, end_offset: int, PAT: str, special_tokens) -> Dict[Tuple[bytes, ...], int]:
    # 局部词频统计
    local_word_freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    # 1. 局部读取文件内容
    with open(file_path, "rb") as f: # 注意：以字节模式 (rb) 打开文件
        f.seek(start_offset)
        # 读取指定范围的字节
        byte_data = f.read(end_offset - start_offset) 
    
    # 2. 将字节数据解码为字符串
    # 假设使用 utf-8 编码
    # 区别：因为file_path使用rb以字节模型打开文件，不会转换 \r\n -> \n, 使用 "r"以utf-8形式打开，就可以自动打开
    chunk_text = byte_data.decode("utf-8", errors="ignore") 
    chunk_text = chunk_text.replace('\r\n', '\n')
    # 这一行也可能需要，以防单独的回车符 '\r' 存在并被分词
    chunk_text = chunk_text.replace('\r', '\n') 
    
    split_pattern = '|'.join(map(regex.escape, special_tokens))
    sub_chunks: List[str] = regex.split(split_pattern, chunk_text)

    # 3. 执行原 process_chunk 的逻辑
    for sub_chunk in sub_chunks:
        for word in regex.finditer(PAT, sub_chunk):
            word_bytes = word.group().encode("utf-8")
            bytes_list_tuple = tuple(bytes([x]) for x in word_bytes)
            local_word_freq[bytes_list_tuple] += 1
        
    return local_word_freq


def parallel_tokenize_and_count_optimized(input_path: str, split_tokens: list, PAT: str) -> Dict[Tuple[bytes, ...], int]:
    # 假设我们只用一个特殊token来做大分割，并且它不是空字符串
    # 将分割token转换为bytes，供 find_chunk_boundaries 使用
    # split_special_token = split_token.encode("utf-8")
    
    # 第1步：找到切分边界 (内存占用低)
    print(f"Finding chunk boundaries for file: {input_path}")
    with open(input_path, "rb") as f:
        # 目标是创建与CPU核心数相同数量的块
        desired_num_chunks = os.cpu_count() or 4
        boundaries = find_chunk_boundaries(f, desired_num_chunks)
        
    # 构造待处理的参数列表 (start_offset, end_offset)
    tasks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        # 确保块不为空
        if end > start:
            tasks.append((input_path, start, end, PAT, split_tokens)) 

    # 第2步：使用进程池并行处理
    final_word_freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    
    # 由于 pool.map 只能接受一个迭代器，而我们需要传入多个参数，
    # 建议使用 pool.starmap 或 functools.partial/lambda
    
    # 使用 pool.starmap (推荐)


    # 包装参数
    args_for_starmap = [(task[0], task[1], task[2], task[3], task[4]) for task in tasks]

    with Pool(processes=len(tasks)) as pool: # 进程数最多不超过tasks数量
        # starmap 接受一个函数和一组参数元组列表
        list_of_local_freqs = pool.starmap(process_file_chunk, args_for_starmap)
        
    # 第3步：合并所有局部词频统计结果
    for local_freq in list_of_local_freqs:
        for token, count in local_freq.items():
            final_word_freq[token] += count
            
    return final_word_freq


def bpe(input_path: str, vocab_max_size: int, special_tokens: list[str]=['<|endoftext|>'], num_processes=16) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    

    # Step 2 统计 word freq
    # 根据special 拆开文本为 多个chunks
    # 分成不同的chunks
    # with open(input_path, "rb") as f:
    #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    #     for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc='Pre-Token', unit='chunks', ncols=100):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         documents = regex.split('|'.join(map(regex.escape,special_tokens)), chunk)
    #         for doc in tqdm(documents, desc="Process chunk", ncols=80, leave=False, unit="docs"):
    #             for word in regex.findall(PAT, doc):
    #                 word_bytes = word.encode('utf-8')
    #                 word_byte_tuple = tuple(bytes([i]) for i in word_bytes)
    #                 word_freq[word_byte_tuple] += 1
    # 换成多线程处理
    # word_freq = build_vocab_parallel(input_path, num_processes, special_tokens, boundaries)                    
    # with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    #     text = f.read() # 读取整个文件内容
    # # # 第3步对语料库里的文段进行预分词pre-tokenization：分割文本时保存标点和空格，得到“单词”列表['Hello', ',', ' world', '!', ' This', ' is', ' a', ' test', '.']
    # chunks = regex.split('|'.join(map(regex.escape,special_tokens)),text) #首先按照特殊字符进行大分割，比如<endoftext>按照章节分割
    # # 然后在大分割里小分割，按照空格和标点
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # for chunk in chunks:
    #     for word in regex.findall(PAT, chunk):
    #         word_bytes = word.encode("utf-8") #对每一个单词进行编码，并转换为bytes
    #         bytes_list = [bytes([x]) for x in word_bytes] #e.g. ['h', 'e', 'l', 'l', 'o']
    #         word_freq[tuple(bytes_list)] += 1 #统计每个token出现的频率
    start_time = time.time()
    word_freq = parallel_tokenize_and_count_optimized(input_path, special_tokens, PAT)
    print(f'pretokenzation running time is {time.time() - start_time: .2f}s')
    
    # Step 2 根据 word freq 统计 pairs freq
    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pairs_count[(word[i], word[i + 1])] += freq
    

    # Step 3 获取 max_pair, 更新vocab dict, merges, 更新 word freq, 更新 pairs freq
    # while vocab_size < vocab_max_size:
    for _ in tqdm(range(vocab_max_size - vocab_size), desc="Merge Pairs...", unit="words", ncols=80):
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
    start_time = time.time()
    input_path  = r"/home/music/wzl/AI-Learning/CS336/data/owt_train.txt"

    vocab_dict, merges_list = bpe(input_path=input_path,
        vocab_max_size=32000, num_processes=32)

    vocab_serializable = {k: v.decode("utf-8", errors="replace") for k, v in vocab_dict.items()}
    merges_serializable = [(i[0].decode("utf-8", errors='replace'), i[1].decode("utf-8", errors="replace")) for i in merges_list]

    max_len = 0
    max_token = ""
    for v in vocab_serializable.values():
        if len(v) > max_len:
            max_len = len(v)
            max_token = v
    print("the max length token is ", max_token)

    # 用于可视化
    with open("owt_train-vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=4) # 美化输出
    with open("owt_train-merge_pair.json", 'w', encoding='utf-8') as f:
        json.dump(merges_serializable, f, ensure_ascii=False, indent=4) # 美化输出
    
    # 保存词汇表到文件 (使用 pickle)
    with open("owt_train-vocab.pkl", "wb") as f:
        pickle.dump(vocab_dict, f)
    
    # 保存合并操作记录到文件 (使用 pickle)
    with open("owt_train-merge_pair.pkl", "wb") as f:
        pickle.dump(merges_list, f)

        
    print(f'running time is {time.time() - start_time: .2f}')
    