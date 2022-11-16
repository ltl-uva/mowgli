from collections import defaultdict
from mowgli.helpers import file2list


class RawTextDataset:
    """Loads text sentence strings from disk into memory."""
    def __init__(self, path: str, languages: list, reduce_size: int = None):
        self.path = path
        self.raw_text_data = defaultdict(dict)

        files = [f"{path}.{l}" for l in languages]
        raw_sentences = [file2list(f, reduce_size=reduce_size) for f in files]

        idx = 0
        for sentence in zip(*raw_sentences):
            for language, sent in zip(languages, sentence):
                self.raw_text_data[idx][language] = sent
            idx += 1

    def __len__(self):
        """Returns length of dataset."""
        return len(self.raw_text_data)

    def __getitem__(self, idx: int) -> dict:
        """Returns element at index `idx`."""
        return self.raw_text_data[idx]

    def __setitem__(self, idx: int, value: dict):
        """Changes element at index `idx` into `value`."""
        self.raw_text_data[idx] = value

    def __iter__(self):
        """Iterates over dataset."""
        for idx in range(len(self.raw_text_data)):
            yield self.raw_text_data[idx]

    @classmethod
    def splits(cls, cfg: dict) -> dict:
        """Creates train, validation and test splits."""
        langs = {}
        langs["train"] = cfg["src"] + cfg["trg"]
        langs["valid"] = cfg["valid_src"] + cfg["valid_trg"] if cfg.get("valid_src") and cfg.get("valid_trg") else langs["train"]
        langs["test"]  = cfg["test_src"]  + cfg["test_trg"]  if cfg.get("test_src")  and cfg.get("test_trg")  else langs["train"]

        datasets = {}
        for split in langs.keys():
            path = cfg.get(split+"_path")
            if not path: continue
            data = cls(path=path, languages=langs[split], reduce_size=cfg["reduce_size"])
            assert len(data) > 0
            datasets[split] = data

        return {
            "train":    datasets["train"] if cfg.get("train_path") else None,
            "valid":    datasets["valid"] if cfg.get("valid_path") else None,
            "test":     datasets["test"]  if cfg.get("test_path")  else None,
        }
