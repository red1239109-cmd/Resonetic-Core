“This module is used internally by Resonetics, DRE, and other systems —
but it is intentionally shipped as a standalone decision primitive.”

This module governs how an existing belief state should be updated
when new observations arrive.

Rather than directly overwriting beliefs, it first decides
*whether* and *how strongly* an update should occur,
based on coherence, shock, and observation quality.

It contains no model, no inference logic, and no domain assumptions.
Only the decision logic for belief updates.
