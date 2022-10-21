from transformers import BertModel
class MyBertModel(BertModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_amr=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        batch_size = input_shape[0]
        if is_amr is not None:
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = self.embeddings.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            amr_pos_shift = is_amr.to(position_ids.dtype) * 128
            position_ids = position_ids + amr_pos_shift.view(-1, 1)
            token_type_ids = token_type_ids + 1

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
