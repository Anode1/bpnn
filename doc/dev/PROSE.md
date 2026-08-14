# How the documents are written

The C style is in `STYLE.md`. This file is the same contract for the prose: the README, the
manual, the reference documents and the commit subjects.

It exists because the documents had drifted into one voice. A README that argues, a reference
table that argues, and a benchmark sheet that argues all read as the same essay, and the same
fact appeared in three files in three phrasings that had stopped agreeing. Both problems have the
same fix.

## The rules

**1. Cut the decorative antithesis.** The construction is `X, not Y` / `rather than` / `instead
of` / `it is not a Z`. It was running at one every three sentences across the tree. Where the
negative half carries no information, assert the positive and stop:

    before   Read the floor as a rough scale, not as a test.
    after    The floor is a rough scale.

    before   the escalation happens on a measurement rather than on preference
    after    the escalation happens on a measurement

Keep it where the wrong reading is one a reader will actually reach for and the negative half
tells them something: `-e` *is* a ceiling and not a count, and a reader who assumes otherwise
misreads the `epochs` column. Those are load-bearing and stay. The five documents run at about
one per 350 words after the cull, down from one per 190, and nearly all of what is left is
load-bearing. Do not chase a number here; chase the decorative ones.

**2. Sections end when the information ends.** No closing maxim. If the last sentence of a
section generalises rather than informs, delete it.

**3. Titles are nouns.** `Memory`, not `Memory, and files larger than it`. Titles are navigation.
The argument goes in the body.

**4. Bold lead-in labels are rationed.** `**What it does.**` opening a paragraph is a chat
habit. Two or three in a README where they mark a real turn; none in a reference document, where
a table or a subheading does the job.

**5. Register follows purpose.** The README persuades and may have voice. `DIAGNOSTICS.md`,
`BENCHMARKS.md` and `STYLE.md` are references: declarative sentences, tables, no rhetoric. A
reference document should be duller than the README. If every document in the project is
interesting, the project reads as though one narrator wrote all of it in an afternoon.

**6. Each fact has one home.** A number lives in the document that owns it, and the others link.
Restating it is how `--cache` came to be documented at two different speeds in two files, and
how the check count was 160 in one paragraph and 165 forty lines below.

**7. Leave things unqualified.** Not every sentence needs its objection answered in the same
breath. Some assertions can stand bare and some questions can go unanswered.

## What is deliberately not on this list

Em-dashes, semicolons and colons. 618 of the dashes in this tree are option flags. The
punctuation was never the signal.

Sentence length. The variation is already wide (mean 24 words, sd 19 in the README) and nothing
should be done to flatten it.

Vocabulary. There is no banned-word list here.

## Checking

Two of these are countable. Over `README.md`, `AGENTS.md` and `doc/`:

    grep -hE '^\*\*[^*]+\*\*' $FILES | wc -l        # rule 4: was 33, now 6
    grep -hE '^#{2,3} .*(, and|, not )' $FILES      # rule 3: was 7, now 1

Read what they print rather than trusting the count. Of the six the first catches, two are inline
emphasis that happens to fall at the start of a line and three mark a real turn. The one the
second catches is the README's subtitle, which is a tagline and not navigation.

The rest are read, not counted.
