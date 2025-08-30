Refactor @backend/ai_generator.py to support sequential tool calling where system can make up to 2 tool calls in separate API rounds.

Current behavior:

- Claude makes 1 tool call → tools are removed from API params → final response
- If system wants another tool call after seeing results, it can't (gets empty response)

Desired behavior:

- Each tool call should be a separate API request where system can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when information from different courses/lessons is needed


Example flow:

User: "Search for a course that discusses the same topic as lesson 4 of course X"
system: get course outline for course X → gets title of lesson 4
system: uses the title to search for a course that discusses the same topic → returns course information
system: provides complete answer


Requirements:

Maximum 2 sequential rounds per user query
Terminate when: (a) 2 rounds completed, (b) system's response has no tool_use blocks, or (c) tool call fails
Preserve conversation context between rounds
Handle tool execution errors gracefully

Notes:

update the system prompt in @backend/ai_generator.py
update the test @backend/tests/test_ai_generator.py
Write tests that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details.


Use two parallel subagents to brainstorm possible plans. Do not implement any code.