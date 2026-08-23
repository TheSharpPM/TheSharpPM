# CLAUDE.md

This file gives Claude Code project-level instructions for **The Sharp PM website** (tspm-website).

## Git identity

All commits in this repository must be authored and committed as:

- Name: `TheSharpPM`
- Email: `thesharppm@gmail.com`

Always pass the identity explicitly on the commit command, even if local repo config already matches:

```
git -c user.name=TheSharpPM -c user.email=thesharppm@gmail.com commit -m "..."
```

Do this every time, regardless of local config state — local config has previously been reset, lost on a re-clone, or overridden, which let commits leak through under a personal email.

Do not add `Co-Authored-By` trailers to commit messages.

## Language

English only, everywhere in this repository: code, comments, commit messages, commit bodies, documentation, and workflow/step names. Portuguese is not allowed anywhere in the repo.
