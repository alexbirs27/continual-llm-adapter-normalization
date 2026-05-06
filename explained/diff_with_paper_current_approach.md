The paper uses T5-large (encoder-decoder). Our model is Qwen2.5-1.5B (decoder-only). This changes everything downstream.
                                                                                                                                                      T5 (paper):                          Qwen2.5 (ours):
  ┌─────────────┐                      ┌──────────────────────────────┐                                                                             
  │   ENCODER   │ ← bidirectional      │  Classify: Arsenal win...    │
  │  (reads     │   attention          │  Label: Sports               │
  │   full      │                      │  ↑                           │
  │   input)    │                      │  causal attention only       │
  └──────┬──────┘                      │  (each token sees only left) │
         │                             └──────────────────────────────┘
  ┌──────▼──────┐
  │   DECODER   │ → generates "Sports"
  │  (produces  │
  │   output)   │
  └─────────────┘

  ---
  Same Example, Both Formats

  Paper (T5, instruction tuning):
  Encoder input:
    "What is the topic of the following paragraph?
     Option: World, Sports, Business, Technology
     Arsenal win Premier League title in dramatic final"

  Decoder target:
    "Sports"

  Ours (causal LM):
  Single sequence (one model, causal attention):
    "Classify: Arsenal win Premier League title in dramatic final
     Label: Sports"

  Loss only on: " Sports"

  ---
  What "Instruction Tuning" Actually Means Here

  The paper's format differs from ours in three concrete ways:

  1. Explicit task definition — tells the model what to do:
  - Paper: "What is the topic of the following paragraph?"
  - Ours: "Classify:" — much shorter, no explanation

  2. Explicit options — tells the model what the valid outputs are:
  - Paper: "Option: World, Sports, Business, Technology" — listed in every example
  - Ours: options are never shown to the model during training or eval

  3. Bidirectional encoder — the encoder reads the entire instruction+text before generating anything. Our causal model reads left-to-right, so when
   it's at "Label:", it can only attend to what came before — it never sees "Label: Sports" as context when processing the article.

  ---
  Training Loss — Subtle but Important Difference

  ┌─────────────────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────────────┐
  │                         │                 Paper (T5)                  │                        Ours                         │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Loss computed on        │ All tokens the decoder generates ("Sports") │ Only the single label token                         │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Eval method             │ Generate text → match string                │ Argmax over 4 label token logits at one position    │
  ├─────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Options visible at eval │ Yes (in input)                              │ No — we manually restrict logits to valid label IDs │
  └─────────────────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────────────┘

  In the paper the model learns to generate the label string. In our setup the model learns to assign probability to the label token at a specific  
  position. At eval we artificially constrain it to valid labels — this is actually a stronger guarantee (the model can't output garbage) but it's  
  not what instruction tuning is designed for.

  ---
  Why This Matters for Your BWT Results

  The paper's richer instruction format has a practical benefit for continual learning: the explicit options in the prompt act as a soft reminder of
   the task structure at every step. When the model forgets a past task, the options in the prompt still steer it toward the right output space.    

  Our format gives the model no such cue — "Classify:" and "Label:" alone don't tell the model which label set is in play, so forgetting the        
  task-specific adapter weights has a bigger impact on performance.