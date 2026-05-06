Let me walk through a concrete ag_news example end to end.

  ---
  Step 1: _format_example — Input

  example = {
      "text": "Arsenal win Premier League title in dramatic final",
      "label": 1   # integer index
  }
  label_names = ["World", "Sports", "Business", "Technology"]
  max_length = 256

  ---
  Step 2: Build Prompt and Target Separately

  text       = "Arsenal win Premier League title in dramatic final"
  label_idx  = 1
  label_name = "Sports"
  target     = " Sports"   # leading space — Qwen tokenizes " Sports" as one token

  The leading space matters: "Sports" and " Sports" tokenize differently in BPE. " Sports" is one clean token; "Sports" might split.

  ---
  Step 3: Truncate the Prompt to Guarantee the Label Fits

  target_token_len = 1          # " Sports" → 1 token
  prompt_budget    = 256 - 1 - 2 = 253

  prompt = "Classify: Arsenal win Premier League title in dramatic final\nLabel:"
  # tokenize with max_length=253 → fits easily for this short example
  prompt_truncated = "Classify: Arsenal win Premier League title in dramatic final\nLabel:"

  For a very long article this step would truncate the article text while keeping "\nLabel:" intact.

  ---
  Step 4: Tokenize the Full String with Padding

  full_text = "Classify: Arsenal win Premier League title in dramatic final\nLabel: Sports"

  # Result (approximate token positions):
  # pos 0:  <bos>
  # pos 1:  "Classify"
  # pos 2:  ":"
  # pos 3:  " Arsenal"
  # ...
  # pos 12: "Label"
  # pos 13: ":"
  # pos 14: " Sports"    ← the label token
  # pos 15–255: <pad>

  input_ids      = [bos, "Classify", ":", " Arsenal", ..., "Label", ":", " Sports", pad, pad, ...]
  attention_mask = [1,   1,          1,   1,           ..., 1,       1,  1,         0,   0,   ...]

  ---
  Step 5: Build the Labels Tensor

  # Re-tokenize just the prompt to find prompt_len
  prompt_only = "Classify: Arsenal win Premier League title in dramatic final\nLabel:"
  prompt_len  = 14   # positions 0–13

  labels = copy of input_ids
  labels[0:14]  = -100   # mask entire prompt
  labels[15:] where attention_mask==0 = -100   # mask padding

  # Final labels:
  # [-100, -100, -100, -100, ..., -100,  token_id(" Sports"),  -100, -100, ...]
  #   0     1     2     3   ...   13           14               15    16

  Only position 14 has a real token ID. The cross-entropy loss trains only on that one position.

  ---
  At Training Time

  The model receives input_ids and labels. PyTorch's causal LM loss computes:

  loss = -log P(token(" Sports") | <bos>, "Classify", ..., "Label", ":")

  The model is explicitly trained to predict " Sports" immediately after "Label:". One gradient step per example, one token of signal.

  ---
  At Eval Time (continual_trainer.py:148–163)

  # Find where the label token is
  target_positions = (label_row != -100).nonzero()[0]
  # → [14]

  # Look at logits one position EARLIER
  pred_pos = 14 - 1 = 13   # position of ":" in "Label:"

  In causal LM the logit at position i predicts token i+1. So:

  - logit at pos 13 (":") → predicts what comes at pos 14 → this is exactly the label prediction

  pred_logits  = logits[i, 13]          # shape: (vocab_size,)
  label_logits = pred_logits[[token_id(" World"),
                               token_id(" Sports"),
                               token_id(" Business"),
                               token_id(" Technology")]]
  predicted = label_logits.argmax()     # → 1 = "Sports" ✓

                                                                                                                
  - logit at pos 13 (":") → predicts what comes at pos 14 → this is exactly the label prediction
     
  pred_logits  = logits[i, 13]          # shape: (vocab_size,)
  label_logits = pred_logits[[token_id(" World"),
                               token_id(" Sports"),                         
                               token_id(" Business"),                 
                               token_id(" Technology")]]                      
  predicted = label_logits.argmax()     # → 1 = "Sports" ✓
                                                                                                                                                    
  # → [14]

  # Look at logits one position EARLIER
  pred_pos = 14 - 1 = 13   # position of ":" in "Label:"

  In causal LM the logit at position i predicts token i+1. So:

  - logit at pos 13 (":") → predicts what comes at pos 14 → this is exactly the label prediction

  pred_logits  = logits[i, 13]          # shape: (vocab_size,)
  label_logits = pred_logits[[token_id(" World"),
                               token_id(" Sports"),
                               token_id(" Business"),
                               token_id(" Technology")]]
  predicted = label_logits.argmax()     # → 1 = "Sports" ✓

  The model never generates text — we just read off one logit position and compare the 4 candidate token scores.

  ---
  Summary

  Training signal:   P(" Sports" | "...Label:")  — one token, one loss value
  Eval decision:     argmax over {" World", " Sports", " Business", " Technology"}
                     at the logit position that predicts what follows "Label:"

  The train and eval are perfectly aligned: training teaches the model to put probability mass on " Sports" at that exact position, and eval reads  
  that probability mass and picks the winner. The -100 masking is what makes it work — without it the loss would train on the entire prompt too,    
  which would drown out the label signal.