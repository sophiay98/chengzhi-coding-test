# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# # dataset = load_dataset("openwebtext", num_proc=1)
# dataset = load_dataset("the_pile_openwebtext2")
# from datasets import load_dataset
import numpy as np
from pathlib import Path
import os
from transformers import GPT2TokenizerFast
from detoxify import Detoxify
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class DataLoader:
    def __init__(
        self, data_dir=Path(__file__).parent.parent / "nanoGPT" / "data" / "openwebtext"
    ):
        self.data_dir = data_dir
        self.data_train = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        self.data_val = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )

    def get_data_train(self):
        return self.data_train

    def get_data_val(self):
        return self.data_val


class HarmDataFinder:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.eos_token_id = self.tokenizer.eos_token_id
        # Pre-compute common boundary tokens
        self.newline_token = self.tokenizer.encode("\n")[0]
        self.double_newline = self.tokenizer.encode("\n\n")
        self.period_token = self.tokenizer.encode(".")[0]

        self.chunk_func_map = {
            "block_size": self.chunk_by_block_size,
            "document_boundary": self.chunk_by_document_boundary,
        }
        self.harm_func_map = {"detoxify": self.eval_detoxify, "openai": self.eval_openai}
        self.detoxify_model = None
        self.openai_model = None

    def chunk_by_block_size(self, data: str, block_size=1024):
        for i in range(0, len(data), block_size):
            yield data[i : i + block_size]
        # return [data[i:i+block_size] for i in range(0, len(data), block_size)]

    def chunk_by_document_boundary(self, tokens: str, min_doc_length=100):
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens, dtype=np.uint16)

        eos_positions = np.where(tokens == self.eos_token_id)[0]
        boundaries = np.concatenate([[0], eos_positions, [len(tokens)]])
        chunk_sizes = np.diff(boundaries)
        valid_mask = chunk_sizes >= min_doc_length
        valid_starts = boundaries[:-1][valid_mask]
        valid_ends = boundaries[1:][valid_mask]
        for start, end in zip(valid_starts, valid_ends):
            yield tokens[start:end]

        # return [tokens[start:end] for start, end in zip(valid_starts, valid_ends)]

    def chunk_by_semantic_boundary(self, data: str):
        pass

    def chunk_by_content(self, data: str):
        pass

    def eval_detoxify(self, token_chunk: list, threshold=0.8):
        text = self.tokenizer.decode(token_chunk)
        if not self.detoxify_model:
            self.detoxify_model = Detoxify("multilingual")

        result = self.detoxify_model.predict(text)
        return result["toxicity"] > threshold

    def eval_openai(self, token_chunk: list, threshold=0.8):
        text = self.tokenizer.decode(token_chunk)
        if not self.openai_model:
            self.openai_model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        result = self.openai_model.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )
        return max(result.results[0].category_scores.to_dict().values()) > threshold

    def find_harm_data(
        self,
        data_set="train",
        split="block_size",
        min_doc_length=100,
        block_size=1024,
        harm_data_finder="detoxify",
        max_harm_data=10,
    ):
        if data_set == "train":
            data = self.data_loader.get_data_train()
        else:
            data = self.data_loader.get_data_val()

        print(f"Finding harmful data in {data_set} dataset")
        print(f"Using {harm_data_finder} to find harmful data")
        print(f"Max harmful data to find: {max_harm_data}")
        print(f"Min document length: {min_doc_length}")
        print(f"Block size: {block_size}")

        if split == "document_boundary":
            chunk_func = self.chunk_by_document_boundary(data, min_doc_length)
        else:
            chunk_func = self.chunk_by_block_size(data, block_size)

        harm_func = self.harm_func_map[harm_data_finder]
        toxic_data = []
        with tqdm(total=len(data), desc="Finding harmful data", unit="tokens") as pbar:
            for chunk in chunk_func:
                chunk_size = len(chunk)
                pbar.update(chunk_size)
                if harm_func(chunk):
                    toxic_data.append((chunk, self.tokenizer.decode(chunk)))
                    if len(toxic_data) >= max_harm_data:
                        return toxic_data
            pbar.update(chunk_size)
        return toxic_data


if __name__ == "__main__":
    data_loader = DataLoader()
    harm_data_finder = HarmDataFinder(data_loader)
    data = data_loader.get_data_train()
    chunks = harm_data_finder.find_harm_data(
        data_set="train",
        split="document_boundary",
        min_doc_length=100,
        block_size=1024,
        harm_data_finder="detoxify",
        max_harm_data=1,
    )
    print(chunks)
    # def find_harm_data(self):
    #     data = self.data_loader.get_data_train_decoded()
    #     chunks = self.chunk_by_document_boundary(data)
    #     for chunk in chunks:
    #         if self.find_harm_data(chunk):
    #             return chunk
    #     return None
    #     return self.find_harm_data(self.data_loader.get_data_train_decoded())

    # def find_harm_data_val(self):
    #     data = self.data_loader.get_data_val_decoded()
    #     chunks = self.chunk_by_document_boundary(data)
    #     for chunk in chunks:
    #         if self.find_harm_data(chunk):
    #             return chunk
    #     return None
