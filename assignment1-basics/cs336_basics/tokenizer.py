from typing import Iterable
import json
import pickle
import regex
from collections import defaultdict
import mmap
from tqdm import tqdm
import time
import numpy as np


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str]= None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.encode_vocab = {token: index for index, token in vocab.items()}
        self.merges_dict = {pair: index for index, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 返回一个实例
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)        
        
        return Tokenizer(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs
        encode = []
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
            sub_chunks = regex.split(f'({split_pattern})', text) # 包含捕获组, 把special token单独放到返回的列表中
        else:
            sub_chunks = [text]

        for sub_chunk in tqdm(sub_chunks):
            if self.special_tokens and sub_chunk in self.special_tokens:
                encode.append(self.encode_vocab[sub_chunk.encode('utf-8')])
                continue
            for word in regex.finditer(PAT, sub_chunk):
                word_bytes = word.group().encode('utf-8')
                merge_word_bytes = self._merge_word_bytes(word_bytes)
                encode.extend([self.encode_vocab[b] for b in merge_word_bytes])
        return encode        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # return a generator that lazily yields token IDs
        for sub_str in iterable:
            yield from self.encode(sub_str)
    
    
    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text
        decode = b""
        for id in ids:
            decode += self.vocab.get(id, b"U+FFFD")
        return decode.decode('utf-8', errors='replace')
    
    def _merge_word_bytes(self, word_bytes) -> list[bytes]:
        word_tokens = [bytes([w]) for w in word_bytes]
        dict_pairs_startIndex: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
        # init dict pairs: startIndex list
        for i in range(len(word_tokens) - 1):
            dict_pairs_startIndex[(word_tokens[i], word_tokens[i + 1])].append(i)
        while True:
            pairs = dict_pairs_startIndex.keys()
            min_index = float("+inf")
            for pair in pairs:
                index = self.merges_dict.get(pair, float("+inf"))
                if index < min_index:
                    min_index = index
                    merge_pair = pair
            if min_index == float('+inf'):
                break
            # 合并逻辑: dict pairs startIndex, word tokens
            start_list = dict_pairs_startIndex[merge_pair][::-1]
            # 合并
            for start in start_list:
                word_tokens[start] = merge_pair[0] + merge_pair[1]
                del word_tokens[start + 1]
            # 更新新的pairs start index逻辑: TODO: 目前只知道从头开始计算，不知道怎么快速的更新
            dict_pairs_startIndex: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
            for i in range(len(word_tokens) - 1):
                dict_pairs_startIndex[(word_tokens[i], word_tokens[i + 1])].append(i)

        return word_tokens


def save_encoded_file(tokenizer: Tokenizer, input_filepath: str, output_filepath: str):
    # Read the input file
    with open(input_filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    # Encode the entire text file
    encoded_tokens = tokenizer.encode(text)

    # Convert to a numpy array and save to file using np.memmap
    encoded_array = np.array(encoded_tokens, dtype=np.int32)
    
    # Use 'w+' mode to allow read/write access to the file
    mmap_array = np.memmap(output_filepath, dtype=np.int32, mode='w+', shape=encoded_array.shape)
    np.copyto(mmap_array, encoded_array)  # Save the encoded tokens to the file
    mmap_array.flush()  # Ensure the data is written to disk
    
    

if __name__ == "__main__":
    vocab_filepath = "TinyStoriesV2-GPT4-train-vocab.pkl"
    merges_filepath = "TinyStoriesV2-GPT4-train-merge_pair.pkl"
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>"])
    input_path = "/home/music/wzl/AI-Learning/CS336/data/owt_train.txt"
    encode_path = "/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-train.txt"
    output_path = "/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-train.dat"
    save_encoded_file(tokenizer, encode_path, output_path)
    encode_path = "/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-valid.txt"
    output_path = "/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-valid.dat"
    save_encoded_file(tokenizer, encode_path, output_path)
                        
    # ____________________________________ encode dataset to file ____________________________________

    
    
    
    # ____________________________________ test compress ratio ____________________________________
    # text_list: list[bytes] = []
    # start = 0
    # end = 0
    # # with open("/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    # with open(input_path, "r", encoding="utf-8") as f:
    #     with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    #         while start < len(mm):
    #             end = mm.find("<|endoftext|>".encode('utf-8'), start)
    #             text_list.append(mm[start:end])
    #             start = end + len(b"<|endoftext|>")
    #             if len(text_list) > 10:
    #                 break
    # # print(text_list[0])
    # bytes_num = 0
    # token_num = 0
    # for text in text_list:
    #     bytes_num += len(text)
    #     encode_num = tokenizer.encode(text.decode('utf-8'))
    #     token_num += len(encode_num)
    # print(bytes_num / token_num)
    
    # ____________________________________ test throughput ____________________________________
    # start = 0
    # end = 0
    # # with open("/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    # start_time = time.time()
    # with open(input_path, "r", encoding="utf-8") as f:
    #     with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    #         bytes_sum = len(mm)
    #         with tqdm(total=len(mm), desc="encode", unit="bytes") as pbar:
    #             while start < len(mm):
    #                 end = mm.find("<|endoftext|>".encode('utf-8'), start) + len(b"<|endoftext|>")
    #                 if end == -1:
    #                     break
    #                 pbar.update(end - start)
    #                 tokenizer.encode(mm[start:end].decode('utf-8'))
    #                 start = end
    #                 if start > 0:
    #                     break
    # run_time = time.time() - start_time

    # throughput = start / run_time
    # print(throughput)
    # text_bytes = 825 * 1024 * 1024 * 1024
    # print(text_bytes / throughput / 60 / 60)
    
    # ____________________________________ base test ____________________________________
    # for encode_list in tokenizer.encode_iterable(iter_str):
    #     print(encode_list)
    #     print(tokenizer.decode(encode_list))

    # ____________________________________ test_roundtrip_unicode_string_with_special_tokens ____________________________________
    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]


    # _____________________________________________ test_overlapping_special_tokens _____________________________________________
    # tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    # test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in ids]
    # print(tokenized_string)
    # assert tokenized_string.count("<|endoftext|>") == 1

    
    
    # ______________________________________________ test_address_matches_tiktoken ______________________________________________
    