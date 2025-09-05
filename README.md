## What is this?
A 2025 revisit of Andrej Karpathy’s 2022 reproduction of LeCun et al. (1989).
I keep the parameter budget constant and test 2023–2025 training tricks
(optimizers, activations, schedulers, regularizers).

Things that worked:

Dropout: Just changing dropout from 0.25 to 0.35, keeping everything else identical, 
improved our test error to 1.44% with 28 misses.

Weight decay: Andrej left the AdamW weight_decay default , which is 0.01, 
I tried setting it to 0.005 (5e-3) and it gave us another improvement — error rate of 1.30% 
with 26 misses with signs of pretty stable generalization (74th, 78th, and 80th passes 
produced 26 misses in test).

Label smoothing: Adding label smoothing with everything else without changes value of 0.02 
produced error of 1.30% with 26 misses.

GELU activation function with default weight decay and dropout of 0.25 and it produced 
1.44% error with 28 misses. When I changed weight_decay=5e-3 the error rate improved 
to 1.30% with 26 misses.

More details in my Medium post:
https://medium.com/@rakivnenko/a-1989-convnet-whats-changed-since-karpathy-updated-lecun-s-33-year-old-code-3-years-ago-676c861e54a6

Link to the original Karpathy's Github: https://github.com/karpathy/lecun1989-repro

**Attribution:** derivative work of Karpathy’s `lecun1989-repro` (MIT).
See `NOTICE` and `vendor/karpathy-2022/LICENSE`.
