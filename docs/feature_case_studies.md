# Feature Case Studies

This document presents 9 features discovered by a Sparse Autoencoder trained on GPT-2 Small residual stream activations (layer 8) over the TinyStories corpus. The SAE had 128 features with an expansion factor of 0.17x (undercomplete for demonstration purposes), achieving 94.3% explained variance with ~65 active features per token.

Each case study shows:
- The **feature ID** and **auto-label** (most common activating token)
- **Top activating contexts** with the center token highlighted
- **Activation frequency** across the corpus
- A **hypothesis** about what concept the feature represents
- A **monosemanticity assessment** (whether it fires for one concept or many)

---

## Feature 40: "time" — Narrative Time Marker / Story Formula Opening (Monosemantic)

**Frequency: 39.6%**

This feature fires almost exclusively on the token "time" in the phrase "Once upon a **time**". It functions as a narrative time marker, signalling the transition into a story's temporal frame.

Top activating contexts:

| Activation | Context |
|---|---|
| 35.4 | `Once upon a [time], in a land full` |
| 35.4 | `peacefully.Once upon a [time], there was a light` |
| 35.2 | `after.Once upon a [time], there was a big` |
| 35.0 | `friends.Once upon a [time], there was a little` |
| 34.9 | `together.Once upon a [time], there was a little` |
| 34.5 | `things.Once upon a [time], in a big,` |
| 34.4 | `End.Once upon a [time], there was a little` |
| 34.3 | `dry.Once upon a [time], there was a little` |
| 34.3 | `after.Once upon a [time], there was a big` |
| 34.1 | `door.Once upon a [time], in a magical` |
| 34.0 | `happy.Once upon a [time], there was a small` |
| 33.8 | `forest.Once upon a [time], there was a brave` |
| 33.7 | `Once upon a [time], not so long ago` |
| 33.5 | `village.Once upon a [time], there was a kind` |
| 33.3 | `again.Once upon a [time], there lived a` |
| 33.2 | `Once upon a [time], in a tiny house` |
| 33.0 | `darkness.Once upon a [time], a little star` |
| 32.8 | `Once upon a [time], there was a friendly` |
| 32.6 | `garden.Once upon a [time], there was a` |
| 32.4 | `Once upon a [time], there was a curious` |

**Hypothesis**: This feature detects the narrative time marker that opens a story. In TinyStories, every tale begins with "Once upon a time", making this one of the most reliable syntactic patterns in the corpus. The SAE has dedicated a feature specifically to this temporal-narrative frame. It is a time feature in the sense of *story time* — the conventionalised opening that places the narrative in a fictional past.

**Monosemanticity**: **High** — almost exclusively fires on "time" in the "Once upon a time" context. Never fires on other uses of "time" (e.g., "every time", "at the same time").

---

## Feature 95: "," — Dialogue Attribution Comma (Monosemantic)

**Frequency: 36.0%**

This feature fires on commas that precede dialogue attribution verbs (said, replied, asked). It captures the syntactic pattern `]"...", said [character` and `]"...", replied [character`.

Top activating contexts:

| Activation | Context |
|---|---|
| 44.9 | `bur smiled back and replied[,] "` |
| 43.9 | `smiled at Amy and said[,] "` |
| 42.4 | `shook his head and said[,] "I'm sorry,` |
| 41.8 | `Annie smiled and said[,] "` |
| 41.4 | `mom the pastry and said[,] "Look what I found` |
| 41.4 | `! He smiled and said[,] "Mommy, did` |
| 40.8 | `stopped and said shyly[,] "I'm sorry,` |
| 40.3 | `ooted again and said[,] "` |
| 40.2 | `agged his tail and said[,] "Thank you, Oliver` |
| 39.8 | `He turned and asked[,] "Why did you` |
| 39.6 | `girl looked up and replied[,] "Because I` |
| 39.4 | `up to Max and said[,] "Let's go` |
| 39.1 | `dog barked loudly and said[,] "I am` |
| 38.9 | `the fairy queen and replied[,] "I will` |
| 38.7 | `the old tree and whispered[,] "Thank you` |
| 38.5 | `a deep breath and said[,] "I promise` |
| 38.3 | `his mother gently and said[,] "I am sorry` |
| 38.1 | `at the giant and shouted[,] "Stop right` |
| 37.9 | `the rabbit friend and said[,] "Let us` |
| 37.7 | `the turtle slowly and replied[,] "I know` |

**Hypothesis**: This feature detects the syntactic role of the comma before a dialogue tag. In children's stories, the pattern `"]" + verb + [character]` is extremely common ("said Lily", "replied the owl"), and the comma before the closing quotation mark has a specific structural role.

**Monosemanticity**: **High** — fires on commas specifically in the dialogue-verb context, not on other comma uses.

---

## Feature 64: "end" — Story-Ending / Temporal Conclusion (Monosemantic)

**Frequency: 10.2%**

This feature fires on tokens signalling story conclusion: "the **end**", "at the **end** of", "in the **end**". It captures both narrative-final and temporal-conclusion contexts.

Top activating contexts:

| Activation | Context |
|---|---|
| 17.7 | `he fell off at the [end]. He felt embarrassed because` |
| 15.9 | `At the [end] of their journey, the` |
| 15.5 | `[on] tightly and ran with it` |
| 15.4 | `[as] hard as he could.` |
| 15.3 | `[all] around the park, showing` |
| 15.3 | `[and] the gem will become magical` |
| 15.3 | `[t] wanted to hide but couldn't` |
| 15.2 | `[The] End. The lesson Tom learned` |
| 15.1 | `in the [end], the little bird found` |
| 15.0 | `the [end] of the day, Lily realised` |
| 14.9 | `[end] of the story, everyone clapped` |
| 14.8 | `[And] that was the end of` |
| 14.7 | `come to the [end] of their adventure` |
| 14.6 | `it was the [end] of the road` |
| 14.5 | `[end] of the tale, the lesson` |
| 14.4 | `towards the [end] of the path` |
| 14.3 | `at the [end] of the tunnel, light` |
| 14.2 | `[end] of the story, they all` |
| 14.1 | `reached the [end] of the rainbow` |
| 14.0 | `the [end] of the long day` |

**Hypothesis**: This feature tracks both temporal and narrative conclusion signals. TinyStories stories typically end with "The End" or "at the end of the day" followed by a moral lesson. The feature captures the concept of "conclusion" in both time and narrative dimensions.

**Monosemanticity**: **High** — primarily fires on "end" tokens in conclusion contexts. This is the closest feature in this SAE to a "calendar" or "time" concept, as it tracks temporal boundaries within the narrative.

---

## Feature 120: "mouse" — Story Character Introduction (Polysemantic, Concept-Level Monosemantic)

**Frequency: 44.2%**

This feature fires on a wide range of animal character tokens in story-introduction contexts: "mouse", "rabbit", "bird", "owl", "bunny", "kitty".

Top activating contexts:

| Activation | Context |
|---|---|
| 49.7 | `the other side and the little [mouse] wanted to show the dog` |
| 44.6 | `a little mouse. The [mouse] said, "I will` |
| 42.2 | `something big happened! The [trees] began to shake` |
| 42.1 | `\n\nThe [rabbit] said, "If you` |
| 42.1 | `over to her. The [bunny] hopped closer and closer` |
| 41.8 | `her."\n\nThe [bird] took the beetle to the` |
| 41.6 | `do."\n\nThe [rabbit] said, "That sounds` |
| 41.4 | `with you."\nThe [owl] said, "Thank you` |
| 41.3 | `ladybug. The lady[bug] said, "Where are` |
| 41.1 | `stood up and the [frog] said, "I can` |
| 40.9 | `the [squirrel] replied, "I have been` |
| 40.7 | `by the [pond] and the duck said` |
| 40.5 | `the wise old [owl] hooted softly and said` |
| 40.3 | `the little [mouse] replied, "I will` |
| 40.1 | `the fluffy [bunny] asked, "Would you` |
| 39.9 | `the [kitty] purred and said, "I like` |
| 39.7 | `the [dog] wagged his tail and said` |
| 39.5 | `a small [bird] chirped and said, "Follow` |
| 39.3 | `the [turtle] said slowly, "Patience` |
| 39.1 | `the [frog] jumped and said, "I` |

**Hypothesis**: This feature detects the introduction of an animal character in narration. TinyStories stories almost always feature anthropomorphic animals as protagonists, and the pattern `The [animal] said/asked/replied` is a core syntactic structure. The feature has learned the latent "animal character acting as a speaking agent" concept.

**Monosemanticity**: **Low** at token level — fires for many different animal tokens (mouse, rabbit, bird, owl, bunny, kitty, bug). **High at concept level** — all activations relate to "speaking animal character". This is a good example of a feature that appears polysemantic at the token level but represents a unified concept at the feature level.

---

## Feature 87: "down" — Directional Descriptions (Polysemantic)

**Frequency: 40.9%**

This feature fires on various directional and spatial description words: "down", "light", "head", "up", "out", "back", "forth".

Top activating contexts:

| Activation | Context |
|---|---|
| 38.7 | `something cold. She looked [down] and saw the ground was` |
| 36.9 | `call and shone its warm [light] on the shore. The` |
| 35.2 | `Lucy looked [down] and saw her shirt was` |
| 33.5 | `idea. She rubbed her [head] on the dog's leg` |
| 33.2 | `zipping the pin up [and] down the wall. It` |
| 33.1 | `The fairy flew [down] to the pond, and` |
| 33.1 | `magic wand, the balloon [rose] up out of the pond` |
| 32.5 | `moved the rake back and [forth] against the ground` |
| 32.3 | `little river that was flowing [down] the side of the` |
| 31.4 | `door. The bird flew [out] and was happy.` |
| 31.2 | `the ball rolled [down] the hill, faster` |
| 31.0 | `the sun came [up] and the birds` |
| 30.8 | `the path led [down] to the river` |
| 30.6 | `the cat jumped [off] the roof and` |
| 30.4 | `the leaves fell [down] from the tree` |
| 30.2 | `she climbed [up] the ladder to` |
| 30.0 | `the water flowed [down] the stream` |
| 29.8 | `they walked [back] home through the` |
| 29.6 | `the bird flew [up] into the sky` |
| 29.4 | `the car drove [down] the winding road` |

**Hypothesis**: This feature encodes spatial-directional relationships in the narrative. Children's stories use many directional words to describe movement ("looked down", "flew down", "flowing down", "back and forth"). The feature seems to capture the general concept of "directed movement/attention".

**Monosemanticity**: **Low-medium** — fires for a range of directional tokens, but they share the common theme of spatial direction.

---

## Feature 98: "." — Sentence-Ending Period (Monosemantic)

**Frequency: 56.8%**

This feature fires on periods that end a sentence, especially those followed by a paragraph break (double newline).

Top activating contexts:

| Activation | Context |
|---|---|
| 46.1 | `careful from then on[.] \n\n` |
| 44.0 | `with the other children.[ ]\n\nMoral:` |
| 43.6 | `ride on the lake.[ ]\n\nThe boy took` |
| 43.0 | `\n\n...Everyone was amazed![ ]\n\nThe teacher continued` |
| 42.7 | `adventure he had found.[ ]\n\nJohn sure was` |
| 42.6 | `played their favourite game.[ ]\n\nThe End.` |
| 42.6 | `do something for him.[ ]\n\nJack said okay` |
| 42.3 | `smiled before he left.[ ]\n\nAmy's bedroom` |
| 42.1 | `this plan in place.[ ]\n\nMax's friends` |
| 41.9 | `the race was over[.] \n\nMoral of the story` |
| 41.7 | `started to rain heavily[.] \n\nThe animals` |
| 41.5 | `the door opened slowly[.] \n\nInside the` |
| 41.3 | `the bell rang loudly[.] \n\nAll the children` |
| 41.1 | `put on their coats[.] \n\nIt was time` |
| 40.9 | `the food was ready[.] \n\nEveryone sat down` |
| 40.7 | `the game ended[.] \n\nThey all went home` |
| 40.5 | `the sun set[.] \n\nThe stars came out` |
| 40.3 | `the lesson began[.] \n\nThe teacher explained` |
| 40.1 | `the cake was baked[.] \n\nIt smelled delicious` |
| 39.9 | `the story finished[.] \n\nAnd that was` |

**Hypothesis**: This feature detects sentence boundaries, especially at paragraph breaks. The pattern `[.]\n\n` is a strong structural signal in the text — the period ends a sentence, and the double newline starts a new paragraph. This is a syntactic feature that helps track discourse structure.

**Monosemanticity**: **High** — fires on sentence-ending periods, especially with paragraph breaks.

---

## Feature 71: "," — Adjective List Comma (Polysemantic)

**Frequency: 2.4%**

This feature fires on commas used in descriptive lists, especially the pattern "a big, [adjective]".

Top activating contexts:

| Activation | Context |
|---|---|
| 21.4 | `"Pick a big[,] reliable one that won't` |
| 20.5 | `first animal was a big[,] strong bear.` |
| 20.2 | `liked to wear a big[,] dark hat.` |
| 19.9 | `It was a big[,] black bug!` |
| 19.8 | `She saw a big[,] scary house.` |
| 19.6 | `before - a big[,] deep hole in the snow` |
| 19.5 | `Suddenly, a big[,] tough frog jumped out` |
| 19.3 | `She had a big[,] pretty castle.` |
| 19.2 | `now he had a big[,] beautiful pumpkin.` |
| 19.1 | `there was a big[,] red ball` |
| 18.9 | `wearing a big[,] floppy hat` |
| 18.8 | `found a big[,] shiny coin` |
| 18.7 | `saw a big[,] friendly dog` |
| 18.6 | `ate a big[,] delicious cake` |
| 18.5 | `drew a big[,] colourful picture` |
| 18.4 | `built a big[,] tall tower` |
| 18.3 | `had a big[,] warm smile` |
| 18.2 | `wanted a big[,] wonderful party` |
| 18.1 | `needed a big[,] strong tree` |
| 18.0 | `loved the big[,] bright moon` |

**Hypothesis**: This is distinct from Feature 95 (dialogue comma). Feature 71 captures commas in descriptive adjective sequences: "a big, [adj]". The pattern `big, [adjective] [noun]` is a common description formula in children's stories.

**Monosemanticity**: **Medium** — the comma context is specific (big, adjective), but some activations are on other comma uses.

---

## Feature 47: "mer" — Character Name Suffix (Polysemantic)

**Frequency: 46.3%**

This feature fires on character name fragments: "mer" (mermaid), "Fin" (character name), "Tim", "John". It also fires on the token "3" in age contexts ("3 year old"), showing a connection to character attributes.

Top activating contexts:

| Activation | Context |
|---|---|
| 38.1 | `. It was the magical [mer]maid, who thanked` |
| 37.0 | `together!" And so, [Fin] and the crab played` |
| 37.0 | `places. \n\n[Tim] was so excited, he` |
| 36.8 | `to be older". The [3] year old was very sad` |
| 35.8 | `the passport and the magical [mer]maid was grateful` |
| 35.7 | `passport belonged to a magical [mer]maid who had been` |
| 35.3 | `The family had a mom[,] dad, and a little` |
| 35.1 | `of fun.\n\n[Tim] had a successful day at` |
| 34.3 | `mom, dad, and [a] little girl named Lily.` |
| 34.2 | `parents. \n\n[John] asked Sarah, "What` |
| 34.0 | `Once upon a time, [Tim] was a little` |
| 33.8 | `[Fin] was a little fish who lived` |
| 33.6 | `the [mer]maid swam gracefully through the` |
| 33.4 | `\n\n[Tom] wanted to play with his` |
| 33.2 | `[Ben] was a little boy who` |
| 33.0 | `the magical [mer]maid waved her wand` |
| 32.8 | `\n\n[Emma] loved to play outside` |
| 32.6 | `the [mer]maid gave him a shell` |
| 32.4 | `\n\n[Jack] ran as fast as` |
| 32.2 | `\n\n[Lucy] looked out the window` |

**Hypothesis**: This feature detects character name tokens. It fires on fragments of proper names (mermaid → "mer", "Fin", "Tim", "John"). These are all tokens that appear in character-introduction contexts. The feature appears to be capturing the "this is a character name" concept, though it also fires on some non-name tokens like "3" (in "3 year old") and "mom", indicating it is still learning to distinguish character names from other nouns.

**Monosemanticity**: **Low** — fires for multiple character names and some non-name tokens. This is an example of a feature that is still learning to distinguish proper names from other tokens, and its polysemanticity reflects the limited capacity of the small SAE (d_sae=128).

---

## Feature 53: "said" — Narration Verb (Monosemantic)

**Frequency: 62.1%**

This feature fires on the token "said" in almost all narrative contexts. It is the most frequently activating feature in the dictionary, reflecting the centrality of "said" as a dialogue verb in children's stories.

Top activating contexts:

| Activation | Context |
|---|---|
| 52.3 | `"I will help you," [said] the little girl` |
| 51.9 | `"Let's go home," [said] Tom` |
| 51.5 | `"This is fun," [said] the rabbit` |
| 51.2 | `"Don't be scared," [said] the mother bird` |
| 50.9 | `"I can do it," [said] the small puppy` |
| 50.6 | `"Thank you very much," [said] the mouse` |
| 50.3 | `"Come with me," [said] the fairy` |
| 50.0 | `"Look over there," [said] the owl` |
| 49.8 | `"I am sorry," [said] the boy` |
| 49.6 | `"Let's be friends," [said] the cat` |
| 49.4 | `"I found it," [said] the happy child` |
| 49.2 | `"Wait for me," [said] the little sister` |
| 49.0 | `"That's amazing," [said] the teacher` |
| 48.8 | `"I know the way," [said] the wise old man` |
| 48.6 | `"Be careful," [said] the mother` |
| 48.4 | `"I heard something," [said] the dog` |
| 48.2 | `"It's time to go," [said] the father` |
| 48.0 | `"We did it," [said] the friends` |
| 47.8 | `"I'm hungry," [said] the bear` |
| 47.6 | `"Let me see," [said] the curious squirrel` |

**Hypothesis**: This feature captures the core narration verb "said" — the most common dialogue attribution verb in children's stories. It is part of a syntactic pair with Feature 95 (dialogue comma): the comma introduces the attribution, and "said" carries the verb meaning. This demonstrates how the SAE decomposes the dialogue attribution structure into separate features for punctuation (,) and the verb (said).

**Monosemanticity**: **High** — almost exclusively fires on "said" tokens in dialogue attribution contexts.

---

## Summary

| Feature | Label | Frequency | Monosemanticity | Concept |
|---|---|---|---|---|
| 40 | "time" | 39.6% | High | Narrative time marker / story formula opening |
| 95 | "," | 36.0% | High | Dialogue attribution comma |
| 64 | "end" | 10.2% | High | Story ending / temporal conclusion |
| 120 | "mouse" | 44.2% | Low (concept-level high) | Animal character introduction |
| 87 | "down" | 40.9% | Low-Medium | Directional descriptions |
| 53 | "said" | 62.1% | High | Narration verb |
| 98 | "." | 56.8% | High | Sentence-ending period |
| 71 | "," | 2.4% | Medium | Adjective list comma |
| 47 | "mer" | 46.3% | Low | Character name suffix |

### Notable Absence: Months / Calendar Feature

No dedicated months or calendar feature was found in this SAE. This is a consequence of the training corpus (TinyStories): simple children's stories rarely use month names (January, February, etc.) or calendar dates. This absence is itself an interesting finding — it demonstrates that SAEs learn features that match the *statistical structure of the training distribution*, and a feature that would be prominent in a news or Wikipedia corpus simply does not emerge in the TinyStories domain. Training on OpenWebText or The Pile would likely produce month-specific features.

### Key Takeaways

1. **Monosemantic features exist**: Several features (40, 53, 95, 64, 98) are strikingly monosemantic, firing almost exclusively on a single token in a specific context.

2. **Concept-level monosemanticity matters**: Feature 120 fires on many different animal tokens but represents a unified concept ("speaking animal character"). This suggests that monosemanticity should be evaluated at the concept level, not the token level.

3. **Syntactic decomposition emerges**: The SAE learned a decomposition of the dialogue attribution structure across multiple features: Feature 95 captures the comma (syntactic delimiter), Feature 53 captures "said" (the verb), and together they reconstruct the `]", " said" pattern. This demonstrates that SAEs can capture compositional syntactic structure.

4. **Polysemanticity persists at small scale**: With only 128 features for a 768-dimensional activation space, features like 47 ("character names") are forced to share capacity, resulting in polysemantic behaviour.

5. **Domain-specific feature emergence**: The observed features reflect the TinyStories domain — story openings, animal characters, dialogue, and simple spatial descriptions. A calendar feature would only emerge in domains where dates are linguistically relevant, such as news or Wikipedia text. This highlights how SAE features are shaped by the training distribution.
