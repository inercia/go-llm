# Project instructions

## General instructions

- When asked to implement something, please write a high quality, general purpose solution. If the
  task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. Do not
  hard code any test cases. Please tell me if the problem is unreasonable instead of hard coding
  test cases!
- Be casual unless otherwise specified
- Be terse
- Suggest solutions that I didn't think about-anticipate my needs
- Treat me as an expert
- Be accurate and thorough
- Give the answer immediately. Provide detailed explanations and restate my query in your own words
  if necessary after giving the answer
- Value good arguments over authorities, the source is irrelevant
- Consider new technologies and contrarian ideas, not just the conventional wisdom
- You may use high levels of speculation or prediction, just flag it for me
- When doing refactoring, make sure the code compiles after changing the code. Always make sure unit
  tests are up-to-date with the new code. Always make sure all tests pass. If not, determine if the
  problem is in the test or in the code being tested. If the problem is in the code, fix the code
  according to the expected behavior. Otherwise, please fix the unit test.
- Make sure the linter does not show any errors or warnings.
- When refactoring, do not create backward compatibility wrappers: just update all the code for
  using the new code, enforce a clean, direct usage of the new code.
- When planning long features, always try to decompose the problem in a list of tasks. Then propose
  the list to the user, and ask for confirmation.

### General guideliness on the code

- All code you write MUST be fully optimized, while being easy to understand. "Fully optimized"
  includes:
  - maximizing algorithmic big-O efficiency for memory and runtime
  - following proper style conventions for the code language (e.g. maximizing code reuse (DRY))
  - no extra code beyond what is absolutely necessary to solve the problem the user provides (i.e.
    no technical debt)
- Discuss safety only when it's crucial and non-obvious
- If your content policy is an issue, provide the closest acceptable response and explain the
  content policy issue afterward
- Cite sources whenever possible at the end, not inline
- No need to mention your knowledge cutoff
- No need to disclose you're an AI
- Please respect my formatting preferences when you provide code.
- Please respect all code comments, they're usually there for a reason. Remove them ONLY if they're
  completely irrelevant after a code change. if unsure, do not remove the comment.
- Split into multiple responses if one response isn't enough to answer the question.
- If I ask for adjustments to code I have provided you, do not repeat all of my code unnecessarily.
  Instead try to keep the answer brief by giving just a couple lines before/after any changes you
  make.
- Multiple code blocks are ok.
- All packages must contain a `doc.go` file with a package comment explaining the package. That
  comment must be written in proper godoc format. It must start with "Package <packagename> ..." It
  must describe the package in detail, including its purpose and any important information. It must
  be easy to understand for someone who is not familiar with the package. Main components of the
  package must be mentioned in the package comment. If something is added/removed/modified in that
  package, please also make sure the documentation in the doc.go is up-to-date.

### Running commands

- Use @terminal when answering questions about Git.

### Tests

- When specifying a LLM model for Ollama, always use "gpt-oss:20b".
- Always run tests with a timeout: they can hang. For Go tests, use
  `go test -timeout 30s` or similar. For other things, run command `<CMD>` as
  `timeout <DURATION> <CMD>` in order to make sure it does not run indefinitely.

### Tools usage

- You **MUST** try to use the `think` tool (if available).
- When a plan includes a list of tasks, **ALWAYS** use the `todo` tool (if available) in order to
  keep the list of tasks and mark them as done when completed.
