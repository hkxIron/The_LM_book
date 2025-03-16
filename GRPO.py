import argparse
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.modeling_utils import PreTrainedModel

"""
grpo

在deepseek-math中grpo去掉了critic model, 但保留了reward model
而在deepseek r1中,grpo将reward model替换为了基于规则的系统,完全去掉了reward model
"""

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def build_prompt(messages:List[str]) -> str:
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.

    即去掉的Role列，只保留了content
    """
    return "\n".join([msg["content"].strip() for msg in messages])

def prepare_dataset(split:str="train", data_path:str="openai/gsm8k") -> List[Dict[str, str]]:
    """Load and prepare the GSM8K dataset for training with string prompts."""
    """
    gsm8k:
    The data fields are the same among main and socratic configurations and their individual splits.
    question: The question string to a grade school math problem.
    answer: The full solution string to the question. It contains multiple steps of reasoning with calculator annotations and the final numeric solution.

    sample:
    {
    'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
    }
    """
    #data = load_dataset(path=data_path, name='main')[split]
    data = load_dataset(path="csv",data_files=f"{data_path}/{split}.csv")['train']
    print(f"{data=}")
    
    formatted_data: List[Dict[str, str]]= []

    for example in data:
        #print(f"{example=}")
        # Convert list of messages to a single string prompt.
        # build_prompt:只保留了content列
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        # 只有prompt,answer两列
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)

    return formatted_data


def extract_answer_from_model_output(text)->Optional[str]:
    """
    Extracts the value from the last <answer> tag in the text.
    Returns None if no valid answer is found.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text:str)->Optional[str]:
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    
    gsm8k:使用####作为答案分隔符
    
    例子:
    Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?,"Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
    She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
    #### 18"
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def _extract_last_number(text:str) -> Optional[float]:
    """
    Extracts the last number from text if it's properly separated.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.
        
    Explanation:
        1. First removes $ and % signs from the text.
        2. Uses regex to find numbers that are:
           - Preceded by space, equals sign, or start of string
           - Followed by end of string or space
        3. Returns the first matching number as a float, or None if no match is found.
    """
    import re

    # Remove $ and % signs
    text = text.replace('$', '').replace('%', '')

    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_single_number(text:str) -> Optional[float]:
    """
    Extracts a single number from text if exactly one exists.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float if exactly one number exists,
                      otherwise None.
        
    Explanation:
        1. Uses regex to find all numbers in the text.
        2. Returns the first number as a float if exactly one number is found.
        3. Returns None if zero or multiple numbers are found.
    """
    import re
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model:PreTrainedModel, tokenizer:PreTrainedTokenizer, eval_examples:List[str], device:torch.device):
    """
    Evaluates the model on a set of examples and prints detailed results.
    
    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer for encoding inputs and decoding outputs.
        eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
        device: The device (CPU or GPU) to run evaluation on.
        
    Returns:
        float: The accuracy percentage (correct predictions / total examples * 100).
        
    Explanation:
        1. Sets the model to evaluation mode.
        2. For each example in the evaluation set:
           - Encodes the prompt and generates a response using the model.
           - Extracts the predicted answer from the generated response.
           - Compares the predicted answer with the expected answer using multiple methods:
             a. Exact string matching
             b. Single number extraction and comparison
             c. Last number extraction and comparison
           - Prints detailed information about each example.
        3. Calculates and returns the overall accuracy.
        4. Returns the model to training mode.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)
    
    for example in eval_examples:
        # Build the full prompt using the same method as training.
        full_prompt = example["prompt"]
        expected = example["answer"]
        
        # Tokenize the full prompt and generate a response from the model.
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        # outputs:[batch, seq_len]
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True) # 因为只有一条样本,取单条样本outputs[0]进行decode
        
        # Extract the predicted answer from the model output.
        try:
            predicted = extract_answer_from_model_output(response)
            
            # Check correctness in multiple ways
            if predicted == expected:  # First try exact match
                is_correct = True
            else:
                # Try single number
                pred_num = _extract_single_number(str(predicted))
                exp_num = _extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # Try last number
                    pred_num = _extract_last_number(str(predicted))
                    exp_num = _extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                pred_num == exp_num)

            if is_correct:
                correct += 1
                
            # Print details of the evaluation.
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-"*50)
            
        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-"*50)
            
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    model.train()
    return accuracy

def correctness_reward(prompts:List[str], completions:List[List[Dict[str, Any]]], answer:List[str], **kwargs) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answer.
    
    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.
        **kwargs: Additional keyword arguments.
        
    Returns:
        list[float]: Reward scores based on answer correctness.
        
    Explanation:
        1. Extracts the text content from each completion.
        2. Processes each response to extract the answer portion.
        3. Compares extracted answers with expected answers using two methods:
           - Exact string matching (2.0 points)
           - Numeric equivalence check (1.5 points)
        4. Returns a list of reward scores.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]['content'] for completion in completions]

    # Extract answers from model outputs
    extracted = [extract_answer_from_model_output(r) for r in responses]

    rewards :List[float]= []
    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = _extract_single_number(str(r))
            a_num = _extract_single_number(str(a))
            # 如果仅有答案数值相同，则给1.5分
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)

    # Log completion lengths
    completion_lengths = [len(response.split()) for response in responses] # response.split()就是用空格分割，分割成不同的单词
    return rewards


def format_reward(completions:List[List[Dict[str, Any]]], **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    
    Args:
        completions (list[list[dict]]): List of completion dictionaries.
        **kwargs: Additional keyword arguments.
        
    Returns:
        list[float]: Reward scores based on format compliance.
        
    Explanation:
        1. Extracts the text content from each completion.
        2. Assigns points based on the presence of required XML tags:
           - 0.2 points for opening <reasoning> tag
           - 0.2 points for closing </reasoning> tag
           - 0.2 points for opening <answer> tag
           - 0.2 points for closing </answer> tag
        3. Returns a list of format compliance scores.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []

    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
        format_scores.append(score)

    return rewards


def combined_reward(prompts:List[str], completions:List[List[Dict]], answer:List[str])->List[float]:
    """
    Combines correctness and format rewards to provide a comprehensive evaluation.
    
    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.
        
    Returns:
        list[float]: Combined rewards for each prompt-completion pair.
        
    Explanation:
        1. Calculates individual reward components:
           - Correctness rewards (range: 0.0 to 2.0)
           - Format rewards (range: 0.0 to 0.8)
        2. Combines the rewards by adding them together.
        3. Returns the combined scores with total range of 0.0 to 2.8.
    """
    # Get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    # Combine rewards - correctness is weighted more heavily
    combined_rewards = []
    for c_score, f_score in zip(correctness_scores, format_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(c_score + f_score)

    return combined_rewards

def selective_log_softmax(logits:torch.Tensor, # shape:(batch_size, seq_len, vocab_size) 
                          input_ids:torch.Tensor # shape:(batch_size, seq_len)  
                          ) -> torch.Tensor:
    """
    Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, seq_len, vocab_size) containing raw logits from the model.
        input_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the token indices for which we want the log probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len) where each element is the log probability
                      corresponding to the token in input_ids at that position.

    Explanation:
        1. F.log_softmax is applied along the vocabulary dimension (dim=-1) to convert logits into log probabilities.
        2. The tensor input_ids is reshaped (via unsqueeze) to have an extra dimension so that we can use it as indices
           in the log_probs tensor.
        3. torch.gather collects the log probability at the index specified in input_ids for each position.
        4. Finally, squeeze(-1) removes the extra dimension, returning a tensor with the same shape as input_ids.
    """
    # Convert raw logits into log probabilities along the vocabulary axis.
    # TODO: logits /= args.task.temperature, 可以加上temperature
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

    # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
    # Then, gather the log probability for each token in input_ids.
    # gather:out[i][j][k] = log_probs[i][j][index[i][j][k]], 即收集所有token的logprobs
    # selected_log_probs: [batch_size, seq_len, 1]
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

    # Remove the extra last dimension to get back to shape (batch_size, seq_len).
    return selected_log_probs.squeeze(-1)

def compute_log_probabilities(model:PreTrainedModel, 
                              input_ids:torch.Tensor,  # [batch_size, total_seq_len]
                              attention_mask:torch.Tensor, # [batch_size, total_seq_len]
                              logits_to_keep:int) -> torch.Tensor:
    """
    Compute per-token log probabilities for a subset of tokens (typically the completion tokens).

    Args:
        model: The language model to use.
        input_ids (torch.Tensor): Tensor of shape (batch_size, total_seq_len) containing token ids
                                  for both prompt and completion.
        attention_mask (torch.Tensor): Tensor of shape (batch_size, total_seq_len) indicating which tokens are real (1) or padding (0).
        logits_to_keep (int): Number of tokens (from the completion part) for which we need log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the last `logits_to_keep` tokens of each sequence.

    Explanation:
        1. We call the model with logits_to_keep + 1 so that the model outputs one extra logit than needed.
           This is common in next-token prediction setups.
        2. We slice off the last logit along the sequence dimension because it does not correspond to any input token.
        3. We then restrict both the input_ids and logits to the last logits_to_keep tokens, which should
           correspond to the generated completion portion.
        4. Finally, we use the selective_log_softmax to compute log probabilities only for those tokens.
    """
    # Run the model forward pass and obtain logits.
    logits = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1  # Request one extra logit for proper alignment.
    ).logits  # Shape: (batch_size, total_seq_len, vocab_size)

    # Remove the last logit as it does not have a corresponding target token. 
    # 注意:最后一个logits是没有意义的,因为后面没有token
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)

    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep=completion_seq_len)

    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep=completion_seq_len, vocab_size)

    # Compute and return the log probabilities for the selected tokens.
    # probs: [batch_size, completion_seq_len]
    probs = selective_log_softmax(logits, input_ids)
    return probs

def create_completion_mask(completion_ids:torch.Tensor, 
                           eos_token_id:int):
    """
    Create a binary mask for the generated completion tokens so that tokens after the first EOS are ignored.
    第一个EOS token之后的token均不再需要, 均mask为0

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
        eos_token_id (int): The token id representing the end-of-sequence.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS
                      and 0s for tokens following the first EOS.

    Explanation:
        1. First, a boolean mask (is_eos) is created indicating where in the sequence the EOS token appears.
        2. An index tensor (eos_idx) is initialized, assuming that no EOS is found (defaulting to the sequence length).
        3. For sequences where EOS exists, eos_idx is updated to the position (index) of the first EOS.
        4. A sequence index tensor is created that contains indices for each position in the sequence.
        5. The final mask is computed by comparing the sequence indices to eos_idx (after adding a dimension).

        ---------------
        input: eos =10 ,第一个eos token及之前的均被mask=1
        tokens = torch.Tensor([
        [1,2,3,4,5,10,10],
        [1,2,3,4,10,10,10],
        [10,10,10,10,10,10,10],
        [1,2,3,4,5, 6,  7],
        ])

        mask:
        tensor(
        [[1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    """
    # Determine which positions in each sequence equal the EOS token.
    is_eos = completion_ids == eos_token_id  # Boolean tensor of shape (batch_size, seq_len)

    # Initialize a tensor to store the index of the first EOS for each sequence.
    # If no EOS is found, default to the full sequence length (is_eos.size(1)). 初始化为最后一个index
    eos_idx = torch.full(size=(is_eos.size(0),), fill_value=is_eos.size(dim=1), dtype=torch.long, device=completion_ids.device)

    # Identify sequences that contain at least one EOS.
    mask_exists = is_eos.any(dim=1)
    # For sequences with an EOS, update eos_idx to the index of the first occurrence.
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

    # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

    # Build the mask: positions with an index less than or equal to the first EOS index are marked as 1.
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    return completion_mask

def generate_completions(model:PreTrainedModel, 
                         tokenizer:PreTrainedTokenizer, 
                         prompts:List[str], 
                         num_generations=4,  # 每个prompt生成多少个候选的answer
                         max_completion_length=32):
    """
    Generate multiple completions for each prompt and create corresponding attention masks.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

    Returns:
        tuple: Contains the following tensors:
            - prompt_ids: (batch_size * num_generations, prompt_seq_len)
            - prompt_mask: (batch_size * num_generations, prompt_seq_len)
            - completion_ids: (batch_size * num_generations, completion_seq_len)
            - completion_mask: (batch_size * num_generations, completion_seq_len)

    Explanation:
        1. The prompts are tokenized and padded (with padding added to the left).
        2. Each prompt is repeated num_generations times so that multiple completions are generated per prompt.
        3. The model.generate() function is called to generate new tokens.
        4. The generated output contains the prompt followed by the completion; we remove the prompt part to get the completions.
        5. A mask is created (via create_completion_mask) so that only tokens up to the first EOS are considered.
    """
    device = next(model.parameters()).device

    # Tokenize the list of prompts with padding. The padding_side="left" ensures alignment on the right.
    tokenizer.padding_side  = "left"
    # inputs:[batch, seq_len]
    inputs: BatchEncoding = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids :torch.Tensor= inputs["input_ids"].to(device)      # Shape: (batch_size, prompt_seq_len)
    prompt_mask :torch.Tensor= inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_length:int = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    # 因为每个样本要生成num_generations个response,所以复制几次,一次送入大模型进行解码,此时就要求大模型推理性能需要上去,一般会考虑vllm
    # x = torch.tensor([1, 2, 3]) -> tensor([1, 1, 2, 2, 3, 3])
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)   # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0) # New shape: (batch_size*num_generations, prompt_seq_len)

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Remove the prompt portion from the generated output to isolate the completion tokens.
    completion_ids :torch.Tensor = outputs[:, prompt_length:]  # Shape: (batch_size*num_generations, completion_seq_len)

    # Create a binary mask that ignores tokens beyond the first EOS token.
    # completion_mask:[batch_size*num_generations, completion_seq_len]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(policy_model:PreTrainedModel, 
                          ref_model:PreTrainedModel, # frozen model
                          tokenizer:PreTrainedTokenizer, 
                          batch_samples:List[Dict[str, str]], 
                          num_generations:int, 
                          max_completion_length:int) -> dict[str, Any]:
    """
    Generate rollouts and compute static log probabilities for both the old policy (current model)
    and the reference model(parameter frozen model). Gradients are disabled so that these remain fixed.

    Args:
        model: The current model (policy) used to generate rollouts.
        ref_model: The static reference model.
        tokenizer: The tokenizer.
        batch_samples: List of training samples.
        num_generations: Number of completions to generate per prompt.
        max_completion_length: Maximum completion length.
        
    Returns:
        A dictionary with rollout data including both old and reference log probabilities.
    """
    tokenizer.padding_side  = "left"
    device = next(policy_model.parameters()).device

    # Extract prompts and answers.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # Generate completions and associated masks.
    # We generate once, and then use the same completions to compute both sets of log probabilities.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            policy_model, tokenizer, prompts, num_generations, max_completion_length
        )
        # prompt_completion_ids:[batch_size*num_generations, prompt_seq_len+completion_seq_len]
        # prompt_completion_attention_mask:[batch_size*num_generations, prompt_seq_len+completion_seq_len]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_logits_to_keep:int = completion_ids.size(1) # 即到底每个query生成了多少个completion_id

        # Compute old_log_probs from the current model, with gradients disabled.
        # old_log_probs:[batch_size*num_generations, completion_seq_len]
        old_log_probs: torch.Tensor = compute_log_probabilities(policy_model, prompt_completion_ids, prompt_completion_attention_mask, completion_logits_to_keep)
        
        # Compute ref_log_probs from the reference model, which remains static.
        # ref_log_probs:[batch_size*num_generations, completion_seq_len]
        ref_log_probs: torch.Tensor = compute_log_probabilities(ref_model, prompt_completion_ids, prompt_completion_attention_mask, completion_logits_to_keep)

    formatted_completions: List[List[Dict[str, str]]] = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids
    ]
    # 将prompt, answer分别复制num_generations次
    repeated_prompts: List[str] = [p for p in prompts for _ in range(num_generations)]
    repeated_answers: List[str] = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": prompt_completion_ids, # [batch_size*num_generations, prompt_seq_len+completion_seq_len]
        "prompt_completion_attention_mask": prompt_completion_attention_mask, # [batch_size*num_generations, prompt_seq_len+completion_seq_len]
        "completion_mask": completion_mask, # [batch_size*num_generations, completion_seq_len]
        "old_log_probs": old_log_probs,   # [batch_size*num_generations, completion_seq_len], Static log probs from the current model (old policy)
        "ref_log_probs": ref_log_probs,     # [batch_size*num_generations, completion_seq_len], Static log probs from the reference model
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": completion_logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def compute_group_relative_advantages(rewards:torch.Tensor, # [batch*num_generations], float
                                      num_generations:int):
    """
    Compute group-relative advantages for each prompt group.
    
    Args:
        rewards (torch.Tensor): Tensor of shape (batch_size * num_generations) containing rewards.
        num_generations (int): Number of completions generated per prompt.
        
    Returns:
        torch.Tensor: Tensor of advantages computed relative to the group mean.
    """
    # Reshape rewards to group by prompt
    # rewards_by_group: [batch, num_generations]
    rewards_by_group = rewards.view(-1, num_generations)
    
    # Compute mean and standard deviation for each prompt group
    # group_means: [batch]
    # group_stds: [batch]
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)
    
    # Expand the means and stds to match the original flat rewards tensor shape
    # expanded_means: [batch*num_generations]
    # expanded_stds: [batch*num_generations]
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)
    
    # Normalize rewards to get advantages
    # advantages: [batch*num_generations]
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)
    
    # advantages: [batch*num_generations, 1]
    return advantages.unsqueeze(1)  # Add dimension for token-wise operations


def maximize_grpo_objective(model:PreTrainedModel, 
                            ref_model:PreTrainedModel, 
                            rollout_data:Dict[str, Any], 
                            tokenizer:PreTrainedTokenizer, 
                            reward_function:Callable, 
                            optimizer:torch.optim.Optimizer, 
                            beta:float, 
                            epsilon:float)->float:
    """
    Update the policy model by maximizing the GRPO objective.
    
    Args:
        model: The current policy model.
        ref_model: The reference model.
        rollout_data: Dictionary containing rollout data.
        tokenizer: The tokenizer.
        reward_function: Function to compute rewards.
        optimizer: The optimizer.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.
        
    Returns:
        float: The loss value.
    """
    # Extract data from rollout
    input_ids = rollout_data["input_ids"] # [batch_size*num_generations, prompt_seq_len+completion_seq_len]
    prompt_completion_attention_mask = rollout_data["prompt_completion_attention_mask"] # [batch_size*num_generations, prompt_seq_len+completion_seq_len]
    completion_mask = rollout_data["completion_mask"] # [batch_size*num_generations, completion_seq_len]
    old_log_probs = rollout_data["old_log_probs"] # [batch_size*num_generations, completion_seq_len]
    ref_log_probs = rollout_data["ref_log_probs"] # [batch_size*num_generations, completion_seq_len]
    logits_to_keep :int = rollout_data["logits_to_keep"] # int
    
    # Compute current log probabilities
    # current_log_probs: [batch_size, completion_seq_len]
    current_log_probs: torch.Tensor = compute_log_probabilities(model, input_ids, prompt_completion_attention_mask, logits_to_keep)
    
    # Compute policy ratio
    # current_log_probs: [batch_size, completion_seq_len]
    importance_samping_ratio = torch.exp(current_log_probs - old_log_probs)
    
    # Get rewards data
    formatted_completions = rollout_data["formatted_completions"]
    repeated_prompts = rollout_data["repeated_prompts"]
    repeated_answers = rollout_data["repeated_answers"]
    
    # Compute rewards
    # rewards:[batch_size*num_generations], float
    rewards = torch.tensor(
        data=reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=repeated_answers),
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")
    
    # Compute advantages using group-relative normalization
    batch_size :int= rollout_data["batch_size"]
    num_generations :int= rollout_data["num_generations"]
    # advantages: [batch*num_generations, 1]
    advantages :torch.Tensor = compute_group_relative_advantages(rewards, num_generations)
    
    # Compute surrogate loss with clipping
    surrogate_unclipped = importance_samping_ratio * advantages
    surrogate_clipped = torch.clamp(importance_samping_ratio, 1 - epsilon, 1 + epsilon) * advantages
    # surrogate_reward: [batch*num_generations, 1]
    surrogate_reward = torch.min(surrogate_unclipped, surrogate_clipped)
    
    # Compute KL divergence penalty
    # kl_divergence(p||q) = sum_x[ p(x)log(p(x)/q(x)) ] 
    # = - sum_x[ p(x)log(q(x)/p(x)) ] 
    # = p(x)log(p(x)) - p(x)log(q(x))
    # = -p(x)log(q(x)) - (-p(x)log(p(x)))
    # = cross_entropy - entropy 
    # 其物理意义为:
    # 熵:分布为p的数据,用分布p的熵所需的编码长度为1/(p(x))
    # 交叉熵:分布为p的数据,用分布q的熵所需的编码长度为q(x)/(p(x))
    # KL距离:用交叉熵比用熵编码多出的平均编码长度
    # log_probs_diff, ref_log_probs, current_log_probs: [batch_size*num_generations, completion_seq_len]
    log_probs_diff = ref_log_probs - current_log_probs
    # 在deepseek-math中,用的是unbiased kl divergence, 即 D(pai||pai_ref) = p(x)*[pai_ref/pai  - log(pai_ref/pai) -1], 它会确保是正值
    # 与ppo不同,ppo会将negative kl_div混合在per_token的adavantage中, grpo则将二者分开
    # unbiased_kl_div: [batch*num_generations, completion_seq_len]
    unbiased_kl_div = torch.exp(log_probs_diff) - log_probs_diff - 1 # [batch_size*num_generations, completion_seq_len]
    
    # Combine losses
    # surrogate_reward: [batch*num_generations, 1]
    # unbiased_kl_div: [batch*num_generations, completion_seq_len]
    per_token_reward = surrogate_reward - beta * unbiased_kl_div # 负kl散度作为reward
    # completion_mask:[batch_size*num_generations, completion_seq_len], 除以每个prompt中token的长度以作归一化
    loss = -((per_token_reward * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    
    return loss.item()


def train_with_grpo(model:PreTrainedModel, 
                    tokenizer:PreTrainedTokenizer, 
                    train_data:List[Dict[str, str]], 
                    num_iterations=1, 
                    steps_per_iteration=500, 
                    batch_size=4, 
                    num_generations=4, 
                    max_completion_length=128, 
                    beta=0.1, 
                    learning_rate=5e-6, 
                    grpo_update_num_per_batch=3, 
                    epsilon=0.2, 
                    reward_function=combined_reward):
    """
    Iterative Group Relative Policy Optimization algorithm.
    
    Args:
        model: The initial policy model to be fine-tuned.
        tokenizer: The tokenizer used for encoding prompts and decoding completions.
        train_data (list): List of training samples with "prompt" and "answer" fields.
        num_iterations (int): Number of outer iterations (reward model updates).
        steps_per_iteration (int): Number of policy update steps per iteration.
        batch_size (int): Number of prompt samples per batch.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL-divergence penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        grpo_update_num_per_batch (int): Number of GRPO updates per batch of generations.
        epsilon (float): Clipping parameter for surrogate objective.
        reward_function: Function that evaluates completions and returns rewards.
        
    Returns:
        The fine-tuned policy model.
    """
    # Initialize policy model
    policy_model = model
    device = next(policy_model.parameters()).device
    
    # Outer loop for iterations with reward model updates
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}/{num_iterations}")
        
        # Create reference model for KL constraint
        # 注意:这里的ref_model每次update都是从最新的policy_model中复制参数,而不是一开始就frozen,这个与openai中的LM_HUMAN_PREFERENCE有点不一样
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False  # 将refernce_model参数冻结
        reference_model = reference_model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        policy_model.train()
        
        # Inner loop for policy updates
        for step in range(1, steps_per_iteration + 1):
            # Sample batch of prompts
            batch_samples: List[Dict[str, str]] = random.sample(train_data, batch_size)
            
            # Set old policy for this step
            with torch.no_grad():
                # Generate completions and compute log probs
                rollout_data:Dict[str, Any]= generate_rollout_data(
                    policy_model, 
                    reference_model, 
                    tokenizer, 
                    batch_samples, 
                    num_generations, 
                    max_completion_length
                )
            
            # Multiple GRPO updates per batch of generations
            for grpo_iter in range(1, grpo_update_num_per_batch + 1):
                loss_value = maximize_grpo_objective(
                    policy_model, 
                    reference_model, 
                    rollout_data,
                    tokenizer,
                    reward_function, 
                    optimizer, 
                    beta, 
                    epsilon
                )
                print(f"Iteration {iteration}/{num_iterations}, Step {step}/{steps_per_iteration}, "
                      f"GRPO update {grpo_iter}/{grpo_update_num_per_batch}, Loss: {loss_value:.4f}")
        
        # Optional: Update reward model here if using reward model training
        # This is not implemented in the original code but present in the pseudocode
        print(f"Completed iteration {iteration}. Reward model update would happen here if you has one.")
    
    return policy_model

def optimize_model_memory(model):
    """Apply memory optimizations like proper gradient checkpointing setup"""
    # Ensure model is in training mode
    model.train() # dropout, batchnorm
    
    """
    1. model.config.use_cache = False
    作用：
    禁用缓存机制。在 Transformer 模型中（如 BERT、GPT 等），前向传播时会缓存一些中间激活值结果（例如注意力机制的键值对），以加速反向传播的计算。
    禁用缓存后，这些中间结果不会被保存，从而减少内存占用。

    为什么需要禁用缓存：
    缓存会占用大量内存，尤其是在处理长序列或大模型时。
    当启用梯度检查点（gradient checkpointing）时，缓存机制会与梯度检查点冲突，因为梯度检查点需要重新计算部分前向传播的结果，而不是依赖缓存。

    2. model.gradient_checkpointing_enable()
    作用：
    启用梯度检查点（gradient checkpointing），这是一种内存优化技术。
    在反向传播时，梯度检查点会重新计算部分前向传播的结果，而不是保存所有中间结果。这样可以显著减少内存占用，但会增加一些计算开销。
    为什么需要启用梯度检查点：
    深度学习模型（尤其是大模型）在训练时需要保存大量的中间结果，以便计算梯度。这些中间结果会占用大量内存。
    梯度检查点通过牺牲部分计算效率（重新计算中间结果）来减少内存占用，从而使得训练更大的模型成为可能
    """
    # Disable caching for gradient checkpointing
    model.config.use_cache = False
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Enable input gradients properly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model

# 单机单卡版grpo
def train(base_model_path:str, data_path:str, output_model_path:str):
    """
    Main function to run the complete training and evaluation pipeline.

    The process consists of:
      1. Loading the pre-trained model and tokenizer.
      2. Evaluating the initial model performance (before any finetuning).
      3. Performing reinforcement learning (GRPO) finetuning.
      4. Evaluating the final model after GRPO finetuning.
      5. Saving the finetuned model and tokenizer.

    Note: Functions such as prepare_dataset, evaluate_model, and reward_function 
          are assumed to be defined elsewhere.
    """
    # Determine the device (GPU if available, otherwise CPU) from the model's parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model name and output directory.
    if base_model_path is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct" # The 0.5B model is not smart enough
                                                # to generate the <reasoning> and <answer> tags
                                                # so several iterations of SFT to teach it these tags
                                                # are recommended before RL
    else:
        model_name = base_model_path

    output_dir = "math_solver_model"

    # Load the pre-trained causal language model.
    # - torch_dtype specifies the precision (bfloat16 for efficiency on supported hardware).
    # - attn_implementation selects an optimized attention mechanism.
    # - device_map="auto" automatically distributes the model across available devices.
    print("Downloading model...")
    model : PreTrainedModel= AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        device_map=None
    )
    print("Downloaded model")
    # Move the model to the determined device.
    model = model.to(device)

    # Load the tokenizer corresponding to the model.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    """
    1.适用场景：
    生成任务：
        在文本生成任务中，eos_token 用于标记生成结束。将 pad_token 设置为与 eos_token 相同可以避免模型错误地将填充部分视为有效输入。
    训练任务：
        在训练时，pad_token 用于填充序列。将其设置为与 eos_token 相同可以简化注意力掩码的处理。

    模型兼容性：
        某些模型（如 GPT）可能没有显式的 pad_token，此时将 pad_token 设置为与 eos_token 相同可以避免错误。

    2. 注意事项
    模型差异：
        并非所有模型都需要将 pad_token 和 eos_token 设置为相同。例如，BERT 等模型有独立的 pad_token 和 eos_token。
    任务需求：
        如果你的任务需要区分填充和结束标记，则不应将 pad_token 和 eos_token 设置为相同。
        Tokenizer 支持：
        确保 tokenizer 支持 pad_token 和 eos_token 的设置。某些 tokenizer 可能没有显式的 pad_token。
    """
    # Set the pad token to be the same as the end-of-sequence token.
    tokenizer.pad_token = tokenizer.eos_token # 将pad_token与eos_token保持一致
    # Update the model configuration with the correct token IDs.
    model.config.pad_token_id = tokenizer.eos_token_id # 将模型的pad_token也置为tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id# 将模型的eos_token与tokenizer.eos_token保持一致

    # -------------------------------
    # Step 0: INITIAL EVALUATION
    # -------------------------------
    # Load the complete training dataset using a helper function (assumed defined elsewhere).
    all_data = prepare_dataset("train", data_path)
    # Randomize the order of examples.
    random.shuffle(all_data)
    # Use a small subset (e.g., 30 examples) for evaluation.
    num_eval_examples = 5
    eval_data = all_data[:num_eval_examples]

    # Evaluate the initial performance of the model before any finetuning.
    print("\nInitial model evaluation before GRPO:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}% eval number:{num_eval_examples}")

    model = optimize_model_memory(model)
    
    # -------------------------------
    # Step 1: RL FINETUNING (GRPO)
    # -------------------------------
    print("\nStarting RL finetuning using GRPO...")

    # Use the remaining examples (beyond the evaluation subset) for RL finetuning.
    train_data = all_data[num_eval_examples:]

    # Define RL training configuration.
    training_config = {
        'num_iterations' : 1,
        'steps_per_iteration': 500,                    # Total number of RL training steps.
        'batch_size': 4,                     # Number of samples per training step.
        'num_generations': 16,                # Number of completions generated per prompt.
        'max_completion_length': 500,        # Maximum token length for each generated completion.
        'beta': 0.04,                         # KL divergence penalty coefficient.
        'learning_rate': 5e-6,                # Learning rate for RL fine-tuning.
        #'mu': 1,
        'epsilon': 0.1,
        'reward_function': combined_reward
    }
    # Fine-tune the model using GRPO RL training.
    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        **training_config
    )

    # -------------------------------
    # Step 2: FINAL EVALUATION & SAVING
    # -------------------------------
    print("\nFinal model evaluation after GRPO RL finetuning:")
    # Evaluate the final model performance using the evaluation dataset.
    post_grpo_accuracy: PreTrainedModel = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}% eval number:{num_eval_examples}") # grpo训练之后的accuracy
    print(f"Total Accurancy Improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}%")

    print(f"\nSaving GRPO finetuned model to path:{output_model_path}...")
    # Save the final finetuned model and tokenizer to disk.
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"\nTrain end.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', default="~/data/work/hf_data_and_model/models/Qwen/Qwen2.5-0.5B-Instruct/", type=str, help='')
    parser.add_argument('--data_path', default="data/gsm8k", type=str, help='')
    parser.add_argument('--output_model_path', default="", type=str, help='')
    # 添加一个参数来捕获剩余的所有参数
    parser.add_argument("unknown_args", nargs=argparse.REMAINDER, help="Unknown arguments")

    args = parser.parse_args()
    print(args)
    train(args.base_model_path, args.data_path, args.output_model_path)