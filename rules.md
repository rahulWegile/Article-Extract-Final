# Newspaper Pipeline - Developer Rules & Guidelines

## 1. Core Architecture Constraints
* **Unbounded Universe:** The pipeline must support an unbounded universe of newspapers. Do not hardcode lists of newspaper names, editions, or languages. Rely on dynamic extraction, Gemini APIs, or spatial/geometric heuristics.
  * **Exception:** `pipeline/intelligence/local/masthead/registry_data.json` holds known newspaper+edition templates as a local, non-AI, OCR-based fast-path for metadata extraction (`pipeline/intelligence/local/masthead/`). `NewspaperClient` (OpenAI by default, Gemini if `LLM_PROVIDER=gemini`) remains the fallback for anything not in the registry or not confidently matched, so overall pipeline coverage stays unbounded — the registry only accelerates/avoids an AI call for repeat, verified papers, it never gates what the pipeline can process.
* **No Unnecessary Dependencies:** Do not introduce new heavy libraries (e.g., PyTorch, new OCR engines). Rely exclusively on the existing stack: `rapidocr`, `pymupdf` (fitz), `opencv-python`, and `pillow`.

## 2. Coding Standards
* **Do Not Write Extra Code:** Write only what is strictly necessary to solve the current problem. Do not over-engineer, create speculative features, or add boilerplate that is not actively used. (YAGNI - You Aren't Gonna Need It).
* **Maintain Existing Patterns:** Follow the established batching and pending-continuation patterns (e.g., 3-page batches for Gemini extraction).
* **Modularity:** Keep functions single-purpose. If you add a new heuristic (like masthead extraction), isolate it in its own helper method.

## 3. AI / Assistant Guidelines
* **Strict Scope:** When modifying the codebase, do not rewrite unrelated pipeline stages.
* **Resource Awareness:** Always account for CPU and RAM limitations when suggesting or implementing new image processing steps.
* **Direct Answers:** Provide concise, direct answers and code snippets without unprompted refactoring.
* **Focus on Functionality:** Do not perform excessive testing or write unnecessary test scripts unless explicitly asked. Focus strictly on completing the requested functionality.
* **Save Tokens:** Avoid long-winded explanations or overly verbose code comments. Keep all interactions and code generations as token-efficient as possible.
* **Fast Task Completion:** Prioritize the quickest, most direct path to solving the user's request. Avoid overthinking edge cases unless they break the core functionality.
