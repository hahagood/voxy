# Review Rules

Use this file only when you need a reminder of what belongs in `llm.custom_terms`.

Keep:

- repeated ASR mistakes that map cleanly to one canonical term
- product names, repo names, file names, and technical terms
- mixed Chinese/English pronunciation errors that recur

Skip:

- sentence rewrites
- summarization artifacts
- punctuation-only edits
- fragments that are not useful standalone terms

Prefer stable mappings such as:

- `open rootot -> OpenRouter`
- `奥拉玛 -> Ollama`
- `giHub -> GitHub`
- `美托ken -> token`

Be cautious with:

- entries shorter than 2 source characters
- mappings that only occurred once
- mappings where the destination is not a standalone term
