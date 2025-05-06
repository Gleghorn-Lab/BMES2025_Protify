import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, AutoModelForSequenceClassification

from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'GLM2-150': 'tattabio/gLM2_150M',
    'GLM2-650': 'tattabio/gLM2_650M',
    'GLM2-GAIA': 'tattabio/gLM2_650M_embed'
}


class GLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.plus_token = "<+>"
        if self.plus_token not in self.tokenizer.vocab:
            print(f"Warning: Token '{self.plus_token}' not found in GLM tokenizer vocabulary.")

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        modified_sequences = [self.plus_token + seq for seq in sequences]
        tokenized = self.tokenizer(modified_sequences, **kwargs)
        return tokenized


class gLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.glm2 = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> torch.Tensor:
        assert not output_attentions or not output_hidden_states, (
            "output_attentions=True and output_hidden_states=True are not supported by gLM2ForEmbedding."
        )

        out = self.glm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state


def get_glm2_tokenizer(preset: str):
    return GLMTokenizerWrapper(AutoTokenizer.from_pretrained(presets[preset], trust_remote_code=True))


def build_glm2_model(preset: str) -> Tuple[gLM2ForEmbedding, AutoTokenizer]:
    model_path = presets[preset]
    model = gLM2ForEmbedding(model_path).eval()
    tokenizer = get_glm2_tokenizer(preset)
    return model, tokenizer


def get_glm2_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
    tokenizer = get_glm2_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.glm
    model, tokenizer = build_glm2_model('GLM2-650')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
