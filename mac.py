from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from torch import nn

class MACGemmaDecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config, memory, layer_idx, n_persistent=6):
        super().__init__(config, layer_idx=layer_idx)
        self.memory = memory
        self.W_Q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        past_key_value = kwargs.get("past_key_values")

        q = self.W_Q(hidden_states)

        h, new_memory_state = self.memory(q, state=getattr(self, "_memory_state", None))

        if past_key_value is not None:
            past_key_value.memory_state = new_memory_state

        hidden_states = hidden_states + h
        self._memory_state = new_memory_state

        return super().forward(hidden_states, *args, **kwargs)
    


