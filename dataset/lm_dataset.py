import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import io
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torch.distributed as dist
from model.model_vlm import MiniMindVLM
import pyarrow.parquet as pq
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#
class VLMDataset(IterableDataset):
    def __init__(self, parquet_path, tokenizer, preprocess, max_length=512,
                 image_special_token='@', shuffle_buffer_size=0, seed=42):

        super().__init__()
        self.parquet_path = parquet_path
        self.stream = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
        self.shuffle_buffer_size = int(shuffle_buffer_size or 0)
        self.seed = int(seed)
        self.epoch = 0
        self._num_rows = None
        if isinstance(parquet_path, str) and os.path.exists(parquet_path):
            try:
                self._num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
            except Exception:
                self._num_rows = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self._num_rows is None:
            raise TypeError('Streaming dataset length is unknown. Please provide a local parquet file with metadata.')
        return self._num_rows

    def create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    @staticmethod
    def _parse_conversations(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    @staticmethod
    def _to_single_image_bytes(value):
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, dict):
            if 'bytes' in value and value['bytes'] is not None:
                return bytes(value['bytes'])
            if 'path' in value and value['path']:
                with open(value['path'], 'rb') as f:
                    return f.read()
        raise TypeError(f'Unsupported image_bytes item type: {type(value)}')

    def _to_image_bytes_list(self, value):
        if isinstance(value, list):
            return [self._to_single_image_bytes(v) for v in value]
        return [self._to_single_image_bytes(value)]

    def _sample_to_tensors(self, sample):
        if 'conversations' not in sample:
            raise KeyError(f"Missing key 'conversations'. Available keys: {list(sample.keys())}")
        if 'image_bytes' not in sample:
            raise KeyError(f"Missing key 'image_bytes'. Available keys: {list(sample.keys())}")

        conversations = self._parse_conversations(sample['conversations'])
        image_bytes = self._to_image_bytes_list(sample['image_bytes'])
        
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_tensor = torch.stack([MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes])
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_tensor

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        total_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id

        stream = self.stream
        if self.shuffle_buffer_size > 0:
            stream = stream.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed + self.epoch)

        for idx, sample in enumerate(stream):
            if idx % total_shards != shard_id:
                continue
            yield self._sample_to_tensors(sample)

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        t = pq.read_table(path); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            ax[i].imshow(Image.open(io.BytesIO(t['image_bytes'][i].as_py()))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{len(t)}条')
