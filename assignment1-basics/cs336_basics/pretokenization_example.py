import os
from typing import BinaryIO
import regex
from collections import Counter
from tqdm import tqdm
import re
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
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

input_path  = r"/home/music/wzl/AI-Learning/CS336/data/TinyStoriesV2-GPT4-valid.txt"
# input_path = r"D:\School\Sustech\AI-Learning\CS336\data\owt_valid.txt"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
special_tokens: list[str]=['<|endoftext|>']
word_freq = Counter()
import psutil, os
import sys, faulthandler
faulthandler.enable()  # 捕捉 C 扩展崩溃
from tqdm.auto import tqdm
print("Memory available:", psutil.virtual_memory().available // (1024**2), "MB")
start_time = time.time()
## Usage
with open(input_path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print("chunks num is", len(boundaries))
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    print(boundaries)
    for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc='Pre-Token', unit='chunks', ncols=100):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        documents = regex.split('|'.join(map(regex.escape,special_tokens)), chunk)

        for doc in tqdm(documents, desc="Process chunk", ncols=80, leave=False, unit="docs"):
            for word in regex.findall(PAT, doc):
                word_bytes = word.encode('utf-8')
                word_byte_tuple = tuple(bytes([i]) for i in word_bytes)
                word_freq[word_byte_tuple] += 1
        
    print('end')
        # Run pre-tokenization on your chunk and store the counts for each pre-token
print(f"running time is {time.time() - start_time:.2f}")