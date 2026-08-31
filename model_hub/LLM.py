import time
import torch
import flashinfer
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()
        temp_hidden_states = self.layernorm(temp_hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim) # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, layer_idx)
        self.kv_cache.sync(layer_idx, start_bdx)
        del query_states, key_states, value_states

        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, dim)
        del temp_attn_out

        # post attention
        residual = hidden_states.clone()

        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        # faster when split batches
        for batch_idx in range(0, bsz, 1):
            # chunk for lower memory comsumption, especially for 1M context
            for start_idx in range(0, seq_len, 65536):
                end_idx = min(seq_len, start_idx + 65536)
                hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :] = self.mlp(hidden_states[batch_idx:batch_idx+1, start_idx:end_idx, :], layer)

        hidden_states += residual
        del residual

        return hidden_states


    # def layer_decode(self, layer_idx, hidden_states):
    #     # print(f'Layer = {layer_idx}')

    #     residual = hidden_states
    #     bsz, seq_len, dim = hidden_states.shape
    #     # assert seq_len == 1, f"Error: seq_len should be 1 for decoding, but got {seq_len}."
    #     layer = self.layers[layer_idx]

    #     hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
    #     query_states, key_states, value_states = self.wqkv(hidden_states, layer)
    #     query_states, key_states = self.position_embedd(query_states, key_states)

    #     query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
    #     key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
    #     value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

    #     key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
    #     attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
    #     hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
    #     hidden_states = residual + hidden_states

    #     residual = hidden_states
    #     hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
    #     hidden_states = self.mlp(hidden_states, layer)
    #     hidden_states = residual + hidden_states

    #     return hidden_states

    def layer_decode(self, layer_idx, hidden_states):
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        # Input RMSNorm, QKV projection, RoPE
        with torch.cuda.nvtx.range("stage/qkv_projection"):
            hidden_states = self.layernorm(
                hidden_states,
                layer.input_layernorm_variance_epsilon,
                layer.input_layernorm_weight,
            )

            query_states, key_states, value_states = self.wqkv(
                hidden_states,
                layer,
            )

            query_states, key_states = self.position_embedd(
                query_states,
                key_states,
            )

            query_states = query_states.view(
                bsz,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            key_states = key_states.view(
                bsz,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            )
            value_states = value_states.view(
                bsz,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            )

        # Append current token's K/V to KV cache
        with torch.cuda.nvtx.range("stage/kv_cache_update"):
            key_states, value_states = self.kv_cache.decode_update_kv_cache(
                key_states,
                value_states,
                layer_idx,
            )

        # RetroInfer sparse attention
        #
        # Wave selection, selected-ID D2H, CPU mapping,
        # execution gather, precise attention 등이 현재는 모두 이 안에 포함됩니다.
        with torch.cuda.nvtx.range("stage/decode_attention"):
            attn_out = self.decode_attention(
                query_states,
                key_states,
                value_states,
                layer_idx,
            )

        # Attention output projection and residual connection
        with torch.cuda.nvtx.range("stage/attention_output"):
            hidden_states = self.wo(
                attn_out,
                layer,
                bsz,
                seq_len,
                dim,
            )
            hidden_states = residual + hidden_states

        # Post-attention norm and FFN
        with torch.cuda.nvtx.range("stage/mlp"):
            residual = hidden_states

            hidden_states = self.layernorm(
                hidden_states,
                layer.post_attention_layernorm_variance_epsilon,
                layer.post_attention_layernorm_weight,
            )

            hidden_states = self.mlp(hidden_states, layer)
            hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device).contiguous()
        for start_bdx in range(0, bsz, self.prefill_bsz):
            end_bdx = min(bsz, start_bdx + self.prefill_bsz)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [prefill_batch_size, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states, self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    # def decode_forward(self, inputs_ids):
    #     hidden_states = self.word_embedding(inputs_ids)

    #     if self.num_gpus > 1:
    #         for ldx in range(self.num_layers):
    #             hidden_states = self.layer_decode(ldx, hidden_states)
    #             hidden_states = self.parameter_move(hidden_states, ldx)
    #         hidden_states = hidden_states.to(self.layers[0].device)
    #     else:
    #         for ldx in range(self.num_layers):
    #             hidden_states = self.layer_decode(ldx, hidden_states)
        
    #     hidden_states = self.layernorm(hidden_states, self.norm_variance_epsilon, self.norm_weight)
    #     logits = self.lm(hidden_states)
        
    #     return logits

    def decode_forward(self, inputs_ids):
        with torch.cuda.nvtx.range("stage/token_embedding"):
            hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):

                # layer 번호를 나타내는 outer range
                with torch.cuda.nvtx.range(f"layer_{ldx:02d}"):
                    hidden_states = self.layer_decode(
                        ldx,
                        hidden_states,
                    )

                    with torch.cuda.nvtx.range("stage/parameter_move"):
                        hidden_states = self.parameter_move(
                            hidden_states,
                            ldx,
                        )

            with torch.cuda.nvtx.range("stage/final_device_move"):
                hidden_states = hidden_states.to(
                    self.layers[0].device,
                )

        else:
            for ldx in range(self.num_layers):

                # 예: layer_00, layer_01, ..., layer_31
                with torch.cuda.nvtx.range(f"layer_{ldx:02d}"):
                    hidden_states = self.layer_decode(
                        ldx,
                        hidden_states,
                    )

        with torch.cuda.nvtx.range("stage/final_norm"):
            hidden_states = self.layernorm(
                hidden_states,
                self.norm_variance_epsilon,
                self.norm_weight,
            )

        with torch.cuda.nvtx.range("stage/lm_head"):
            logits = self.lm(hidden_states)

        return logits


    def sampling(self, logits, do_sample=False, temperature=0.6, top_p=0.95, top_k=20):
        if not do_sample:
            output_ids = logits.argmax(dim=-1)  # [bsz, 1], torch.int64
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [bsz, 1, vocab_size]
            probs = probs.squeeze(1) # [bsz, vocab_size]
            if top_k != 0:
                output_ids = flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, top_p=top_p, top_k=top_k)
            else:
                output_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p=top_p)
            output_ids = output_ids.unsqueeze(1) # [bsz, 1], torch.int32

        return output_ids


    def inference(self, inputs_ids, do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        # Prefilling
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s", 'green'))

        # CUDAGraph Capture (if enabled)
        if self.attention_type == "RetroInfer":
            self.kv_cache.capture_cuda_graph()
        
        # check if get EOS token during decoding
        if not ignore_eos:
            end_of_text = torch.zeros((self.batch_size, 1), dtype=torch.bool, device=inputs_ids.device)
            token_id_dtype = torch.int64 if not do_sample else torch.int32  # flashinfer returns int32
            eos_token = torch.empty((self.batch_size, 1), dtype=token_id_dtype, device=inputs_ids.device).fill_(self.tokenizer.eos_token_id)
        
        # # Decoding
        # print("Start decoding ...")

        # # syko start
        # # Nsight Systems profiling configuration
        # profile_warmup = 5   # 먼저 실행하고 버릴 decode iteration 수
        # profile_steps = 2    # 실제 capture할 decode iteration 수
        # profile_started = False
        # profile_stopped = False

        # # 전체 decode latency 측정 시작 전에 앞선 CUDA 작업 완료
        # torch.cuda.synchronize()
        # # syko end
        
        # decode_start = time.time()

        # for step in range(self.max_new_length-1):

        #     # syko start
        #     # step=0~4까지 warm-up하고, step=5 진입 직전에 profiler 시작
        #     if step == profile_warmup:
        #         torch.cuda.synchronize()
        #         torch.cuda.cudart().cudaProfilerStart()
        #         profile_started = True

        #     # 한 decode iteration 전체에 NVTX 이름 부여
        #     torch.cuda.nvtx.range_push(f"decode_step_{step}")
        #     # syko end

        #     logits = self.decode_forward(inputs_ids=output_ids)
        #     output_ids = self.sampling(logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k)
        #     if not ignore_eos:
        #         end_of_text |= (output_ids == eos_token)
        #         if end_of_text.all():
        #             print(colored("All sequences have reached EOS token, stop decoding.", 'yellow'))
        #             break
        #     outputs_ids.append(output_ids)

        #     # syko start
        #     torch.cuda.nvtx.range_pop()

        #     # step=5, 6 두 iteration이 끝난 뒤 profiler 종료
        #     if step == profile_warmup + profile_steps - 1:
        #         torch.cuda.synchronize()
        #         torch.cuda.cudart().cudaProfilerStop()
        #         profile_stopped = True

        # # EOS 때문에 capture window 이전에 종료된 예외 처리
        # if profile_started and not profile_stopped:
        #     torch.cuda.synchronize()
        #     torch.cuda.cudart().cudaProfilerStop()

        # # 마지막 decode GPU 작업이 실제로 완료될 때까지 대기
        # torch.cuda.synchronize()

        # # syko end

        # decode_end = time.time()
        # Decoding
        print("Start decoding ...")

        # Nsight Systems profiling configuration
        profile_warmup = 20
        profile_steps = 8

        profile_started = False
        profile_stopped = False

        # Prefill, cache initialization 및 CUDA Graph capture가
        # decode latency에 섞이지 않도록 완료시킵니다.
        torch.cuda.synchronize()
        decode_start = time.perf_counter()

        for step in range(self.max_new_length - 1):

            # step 0~4는 warm-up입니다.
            # step 5의 CUDA 작업이 제출되기 직전에 profiler를 시작합니다.
            if step == profile_warmup:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                profile_started = True

            stop_for_eos = False

            # 한 token의 decode-forward와 sampling 전체
            with torch.cuda.nvtx.range(f"decode_step_{step}"):

                logits = self.decode_forward(
                    inputs_ids=output_ids,
                )

                with torch.cuda.nvtx.range("stage/sampling"):
                    output_ids = self.sampling(
                        logits,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )

                if not ignore_eos:
                    end_of_text |= (output_ids == eos_token)

                    # 원래 코드도 Python if에서 GPU 결과를 확인하므로
                    # ignore_eos=False일 때는 synchronization이 발생합니다.
                    stop_for_eos = bool(end_of_text.all().item())

                # 기존 코드와 동일하게 모든 sequence가 EOS에 도달하면
                # 해당 output_ids는 결과에 append하지 않습니다.
                if not stop_for_eos:
                    outputs_ids.append(output_ids)

            # profile_steps=8이면 step 5~12를 capture합니다.
            if step == profile_warmup + profile_steps - 1:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                profile_stopped = True

            # 반드시 decode_step NVTX range가 닫힌 뒤 break합니다.
            if stop_for_eos:
                print(colored(
                    "All sequences have reached EOS token, stop decoding.",
                    "yellow",
                ))
                break

        # EOS 등으로 capture window가 조기에 종료된 경우
        if profile_started and not profile_stopped:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            profile_stopped = True

        # Nsight capture 이후에 실행된 나머지 decode 작업도 완료
        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start), 4)} s ({round((decode_end - decode_start) * 1000 / (len(outputs_ids) - 1), 2)} ms/step), "
            f"Throughput: {round(self.batch_size * (len(outputs_ids) - 1) / (decode_end - decode_start), 2)} tokens/s",
            'green'
        ))

        print(colored(f"End2End Latency: {round((prefill_end - prefill_start + decode_end - decode_start), 4)} s\n", 'green'))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config,
                 do_sample=False, temperature=0.6, top_p=0.95, top_k=20, ignore_eos=True, 
                 prefill_bsz=1, prefill_method="full"):
        """ LLM Inference.
        Args:
            attention_type: str, Full_Flash_Attn or RetroInfer
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config (dict): The deoding attention configuration.
            do_sample, temperature, top_p, top_k, ignore_eos: The sampling parameters.
            prefill_bsz (int): The batch size for prefill.
            prefill_method (str): The method for prefill, support full and xattn.
        """
        self.attention_type = attention_type

        bs, input_length = inputs_ids.shape
        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        assert self.input_length + self.max_new_length <= self.max_length, \
            f"Error: input_length({self.input_length}) + max_new_length({self.max_new_length}) exceeds max_length({self.max_length})"

        # compute valid start position for each sequence
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks

        self.prefill_bsz = min(prefill_bsz, self.batch_size)
        self.prefill_method = prefill_method
        # set prefill batch size to 1 and prefill method to full attention if input sequences are not in the same length
        if not (valid_start == 0).all():
            self.prefill_bsz = 1
            self.prefill_method = "full"

        print("Allocate GPU buffers and CPU pin memory ...")
        self.init_kv_cache(valid_start, attn_config)

        outputs = self.inference(
            inputs_ids, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            ignore_eos=ignore_eos
        )

        return outputs