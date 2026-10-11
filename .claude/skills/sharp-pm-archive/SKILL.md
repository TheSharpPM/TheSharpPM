---
name: sharp-pm-archive
description: Answer questions using The Sharp PM archive - curated PM links from the last 90 days and every published weekly edition. Use for reading recommendations, "has this been covered", finding a contrarian take, pulling team homework on a topic, or checking what the editor has already argued before writing something new.
---

# Researching the Sharp PM archive

## What this archive actually is

Two very different things, and treating them the same is the main way
answers come out weak:

- **~870 curated links** from 90 days across 15 sources, every one
  tagged. This is a *feed*. Anyone could assemble it by subscribing to
  the same sources. It answers "what exists about X".
- **26 published editions**, each with an editorial, key takeaways, a
  contrarian pick, team homework, and a stated reason for every
  must-read. This is *original* - it exists nowhere else. It answers
  "what do we think about X".

When a question is about an opinion, a position, or a recommendation,
the editions are the answer and the links are supporting material. Lead
with the editorial take; use links to back it.

## How to query it

Tools come from the `sharp-pm-archive` MCP server:
`list_tags`, `search_items`, `search_editorials`, `get_edition`.

**Call `list_tags` before filtering.** Tags are capitalised phrases
("Product Strategy", "AI Strategy"), not slugs. A guessed
`"ai-strategy"` matches nothing and returns a confident, empty answer -
which is worse than an error, because it reads like "we never covered
this".

**Reach for `search_editorials` before `search_items`** on any question
containing "should", "what do you think", "is X worth it", or asking for
a position. `search_items` is for "what has been published".

## Answering well

**Always cite the URL.** Every item and must-read carries one. An
unsourced claim about what an article argues is the exact failure the
editorial pipeline has a publish gate to prevent - do not reintroduce
it by hand.

**Quote the `why`, not the title.** Each must-read has a one-line
reason it was chosen. That reason is the value; the title is just a
label.

**Respect the audience field.** Editions target Senior, Staff, or
Lead/Principal PM. If someone names their level, prefer editions
written for it and say which you used.

**Say when the archive is silent.** The window is 90 days for links and
26 editions for opinions. "Not covered" is a real, useful answer.
Inventing a position to seem helpful is the one unrecoverable mistake
here. If `total_matches` is 0, report that plainly and offer the
nearest adjacent topic you did find.

**Don't reproduce a whole editorial** when a reader asks what it said.
Give the take, the takeaways, and the link. The editorial is the
product; summarising it drives people to it, replacing it does not.

## Worked shapes

*"What should I read on pricing as a Staff PM?"*
→ `search_editorials("pricing")` for the position, then
`search_items(query="pricing", content_type="article")` for current
material. Prefer any edition whose `target_audience` is Staff PM. Give
3-5 links, each with its `why`.

*"Have we covered platform gatekeeping?"*
→ `search_editorials("platform gatekeeping")`. If matches, name the
editions, dates, and the stance. If none, say so and check
`search_items` for whether the *links* touched it even though no
editorial did - that distinction is itself a useful answer.

*"Give my team homework on discovery."*
→ `search_editorials("discovery")`, collect `pm_homework` across the
matching editions, and attribute each item to its edition date.
