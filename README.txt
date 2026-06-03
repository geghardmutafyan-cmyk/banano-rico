# RAG folder

Put nutrition reference files here. Supported formats in the current agent:

- `.txt`
- `.md`
- `.csv`
- `.json`

Examples:

- food rules
- meal templates
- macro guidelines
- allowed/disallowed foods
- local cuisine examples
- portion rules

The report agent loads these files and uses them as reference context when creating or revising `{username}_nutrition_report.md`.
