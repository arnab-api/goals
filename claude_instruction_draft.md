recall our discussions about the multi predicate setup. Currently, we have a working implementation of expanding the trees and extracting anchor points (for labeling the p+ and p- spans inside branches).                                               
  1. for each (p+, p-, prefill) combination we get an MCTS expansion. just from looking at some tree expansions for the inconsequential prefill "", I am seeing that the LM almost always chooses to commit to a predicate (either p+ or p-) first. so, we  
  can't contrast them against each other by sampling from the same tree. I think we need do cross-tree pairing (a p+ branch from T1 and a p- branch from another T2, both T1 and T2 are generated with (p+, p-) but with different prefills).
  2. I think even with cross-tree pairings, we might not have enough contrastive pairs to train the instrumental goal directions. In that case we might even need to implement a combination of Option 3.B and Option 3.c


  going back to the multi predicate setup. now that we have a working implementation of extracting the MCTS trees and the anchor tokens based on green branches. the next step is to