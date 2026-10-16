from pathlib import Path

import pytest

from more.data import ByteTokenizer, TextChunkDataset, make_datasets, split_token_ids


def test_utf8_byte_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello — café"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_chunk_dataset_shifts_targets() -> None:
    dataset = TextChunkDataset([1, 2, 3, 4, 5], block_size=3)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [1, 2, 3]
    assert targets.tolist() == [2, 3, 4]


def test_split_validates_fraction() -> None:
    with pytest.raises(ValueError):
        split_token_ids([1, 2, 3], 1.0)


def test_make_datasets_from_file(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    path.write_text("A small deterministic corpus. " * 20, encoding="utf-8")
    train, valid, tokenizer = make_datasets(path, block_size=8)
    assert len(train) > 0
    assert len(valid) > 0
    assert tokenizer.vocab_size == 259
