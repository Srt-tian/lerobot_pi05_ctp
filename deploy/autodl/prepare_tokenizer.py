"""Package the existing OpenPI PaliGemma SentencePiece vocabulary for AutoTokenizer."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import GemmaConverter, GemmaSentencePieceExtractor

source = Path("/root/autodl-tmp/models/paligemma_tokenizer.model")
out = source.parent / "paligemma_tokenizer"
sp = spm.SentencePieceProcessor(model_file=str(source))
assert (sp.pad_id(), sp.eos_id(), sp.bos_id(), sp.unk_id()) == (0, 1, 2, 3)
original = SimpleNamespace(
    vocab_file=str(source),
    pad_token="<pad>",
    eos_token="<eos>",
    bos_token="<bos>",
    unk_token="<unk>",
    add_prefix_space=False,
)


class CachedSentencePieceExtractor(GemmaSentencePieceExtractor):
    def __init__(self, model):
        super().__init__(model)
        # transformers 5.5.4's legacy Gemma extractor still expects this field.
        self.sp = spm.SentencePieceProcessor(model_file=model)


class CachedGemmaConverter(GemmaConverter):
    SpmExtractor = CachedSentencePieceExtractor


backend = CachedGemmaConverter(original).converted()
backend.post_processor = TemplateProcessing(
    single="<bos> $A", pair="<bos> $A $B:1", special_tokens=[("<bos>", 2)]
)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
    padding_side="right",
    model_max_length=200,
)
prompts = [
    f"Task: Put the pens into the pen holder., State: {' '.join(str((i + j) % 256) for j in range(14))};\nAction: "
    for i in range(256)
]
for prompt in prompts:
    assert tokenizer.encode(prompt) == [2] + sp.encode(prompt), prompt
tokenizer.save_pretrained(out)
loaded = AutoTokenizer.from_pretrained(out, local_files_only=True)
assert loaded.encode(prompts[0]) == [2] + sp.encode(prompts[0])
report = {
    "source": "IDC existing OpenPI cache /pfs/user/.cache/openpi/big_vision/paligemma_tokenizer.model",
    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "vocab_size": len(tokenizer),
    "prompt_equivalence_cases": 256,
    "bos_id": 2,
    "eos_appended": False,
}
(out / "provenance.json").write_text(json.dumps(report, indent=2))
print("TOKENIZER_VALIDATED", json.dumps(report), flush=True)
