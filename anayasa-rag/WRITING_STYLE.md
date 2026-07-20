# Writing Style Rules For Daily Logs

These rules apply to every `DayN.md` file the agent writes. The goal is for
the log to read like it was actually written by a computer engineering intern
at the end of a work day, not like a polished article.

## Structure

- Start the day with a short overview, 2 to 3 sentences, saying in general
  terms what got done. This goes right at the top, no heading needed other
  than the day title.
- After the overview, go into the details, what was changed, why, what broke,
  what got fixed.
- Small code snippets can be added when they help show what changed. Keep them
  short, a few lines is enough, do not paste a whole file or a whole function
  if only one part matters.
- End with a short closing line about how the day went overall, this can be
  one sentence.

## Tone And Vocabulary

- Write like an intern, not like a technical writer. Plain words, normal
  sentence length.
- Do not use words like leverage, utilize, robust, seamless, delve, boast,
  streamline. These make it obvious the text was written by AI, use normal
  words instead, "use", "fix", "made it work", "handled".
- Do not use connector words like Furthermore, Moreover, Additionally, In
  conclusion. An intern normally just writes "also", "and", "so", "then".
- Small imperfections in grammar are fine, it does not need to be perfect
  English, it needs to sound like a real person wrote it fast at the end of
  the day.
- Keep sentences a bit informal, contractions like "didn't" or "wasn't" are
  fine.

## Punctuation Rules

- Never use an em dash (—) or en dash (–) anywhere.
- Avoid semicolons, split into two sentences or use "and" / "but" instead.
- Do not use triple dots (...) for dramatic effect.
- Avoid overly balanced structures like "not only X but also Y", just say it
  plainly.
- Stick to periods, commas, and question marks, keep punctuation simple.

## Terminology Rules

- Never explain what a term, tool, or concept means, not even briefly.
- Wrong: "I used RAG (Retrieval Augmented Generation, a technique that
  combines search with generation) to make the answers better."
- Right: "I added RAG to the answer pipeline so it pulls from the docs before
  answering."
- Assume the reader already knows the stack, this is a personal logbook, not
  a tutorial written for someone else.

## Example Day Entry

```
# Day 3

Today I mostly worked on the login endpoint and fixed a bug that was
breaking the token refresh.

In the morning I checked the issues left from yesterday and saw the refresh
token was not getting saved correctly. I looked into the auth service and
found the problem was in how we compared the expiry dates.

if token.expires_at < datetime.utcnow():
    raise TokenExpiredError()

The check above was using the wrong timezone so tokens that were still valid
got marked as expired. Fixed it by making sure both sides use UTC.

After that I added a small test for this case so it does not break again.
Also updated the README with the new env variable we need for the refresh
secret.

In the afternoon I paired with Mert on the frontend side to connect the new
endpoint, took a while because the response shape changed and some
components were still expecting the old one.

Overall a good day, fixed something annoying and got the login flow working
end to end.
```

## Example CHANGES.md Entry

```
## Revision 1 - 2026-07-21
User asked to add more detail about the timezone bug and remove a sentence
about Postman that was not relevant. Updated the second paragraph to explain
the UTC fix a bit more and removed that sentence.
```
