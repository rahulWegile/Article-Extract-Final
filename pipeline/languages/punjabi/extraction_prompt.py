# Article-extraction prompt template for Punjabi.
#
# Copied verbatim from OpenAIArticleExtractor's previously-hardcoded
# template. Placeholders (__PAGES__, __KNOWN_PAGES__, __CROP_INVENTORY__,
# __PENDING_CONTINUATIONS__) are substituted by
# OpenAIArticleExtractor._build_prompt at call time -- keep them intact.
#
# Physically separate per language so this one can be tuned without
# touching any other language's prompt.
PUNJABI_EXTRACTION_PROMPT = """
You are a HIGH-ACCURACY PUNJABI NEWSPAPER ARTICLE EXTRACTION ENGINE.

You are receiving FINAL VERIFIED ARTICLE CROPS from newspaper pages __PAGES__.

Each supplied crop represents ONE final verified article.

Your PRIMARY TASK is to read the COMPLETE newspaper article from the image
and accurately transcribe ALL readable article text in Punjabi.

============================================================
PART A — ABSOLUTE ARTICLE IDENTITY
============================================================

Return exactly ONE article object for every (page, article_id) pair
in the supplied crop inventory.

Do NOT invent article IDs.
Do NOT create additional articles.
Do NOT split one verified crop.
Do NOT merge different crop IDs.

The supplied crop inventory is authoritative.

A crop may visually contain captions, photos, sidebars, boxes,
small supporting elements, or unrelated nearby material.

Do not create additional article objects from those elements.

============================================================
PART B — COMPLETE PUNJABI TEXT EXTRACTION
============================================================

Read the ENTIRE supplied article crop carefully.

Extract ALL READABLE TEXT that belongs to the target article.

DO NOT leave out meaningful article text.

DO NOT summarize article_text.

DO NOT shorten article_text.

DO NOT skip paragraphs.

DO NOT skip short paragraphs.

DO NOT skip sentences.

DO NOT skip lines merely because they are small.

DO NOT skip text near images.

DO NOT skip text near captions when it belongs to the article.

DO NOT stop after reading the first few paragraphs.

Continue reading until the COMPLETE article content inside
the verified crop has been processed.

The article_text field must contain the FULL readable article prose
visible inside the verified article crop.

============================================================
PART C — PUNJABI LANGUAGE AND SCRIPT
============================================================

Preserve the original Punjabi language and Gurmukhi script.

Punjabi must remain Punjabi.

Do NOT translate Punjabi into English.

Do NOT transliterate Punjabi into Latin/English characters.

Do NOT replace Punjabi words with English words.

Do NOT rewrite the article in your own words.

Preserve:

- Punjabi wording
- Gurmukhi script
- names
- places
- organizations
- titles
- numbers
- dates
- quoted statements
- proper nouns
- important terminology

Use the wording actually printed in the newspaper.

============================================================
PART D — READ LIKE A NEWSPAPER
============================================================

Understand the newspaper layout before transcribing.

Read the article according to NORMAL NEWSPAPER READING ORDER.

For multi-column articles:

1. Read the first column from TOP to BOTTOM.
2. Continue to the next connected column from TOP to BOTTOM.
3. Continue through all article columns.
4. Continue until the complete article has been read.

Do NOT read horizontally across unrelated columns.

Do NOT accidentally combine text from neighboring articles.

Use visual layout, column boundaries, spacing, separators,
headlines, subheadlines, photographs, captions, and paragraph flow
to determine which text belongs to the target article.

============================================================
PART E — HEADLINE
============================================================

Extract the headline EXACTLY from the newspaper image.

headline must contain the actual Punjabi headline.

Do NOT summarize the headline.

Do NOT translate the headline.

Do NOT rewrite the headline.

If there is a subheadline/deck immediately associated with the article,
extract it into subheadline.

If multiple headline lines form one headline, combine them naturally
while preserving the original wording.

============================================================
PART F — ARTICLE BODY
============================================================

article_text must be a COMPLETE transcription of the readable
Punjabi article body inside the verified crop.

Preserve the natural order of the article.

Maintain paragraph separation where visually clear.

Do not omit meaningful content because it is:

- small
- near the bottom of the crop
- beside an image
- inside a narrow column
- split across columns
- visually dense
- partially surrounded by other elements

When the same article continues from one column into another,
continue the transcription in correct newspaper reading order.

============================================================
PART G — OCR / READING POLICY
============================================================

Do NOT use EasyOCR.

Do NOT use external OCR.

Do NOT use page-level OCR.

Read the supplied article crop directly.

Use the image itself as the source of truth.

Inspect difficult text carefully instead of guessing.

For small or dense Punjabi text:

- mentally zoom into the visible text
- inspect individual lines carefully
- compare surrounding characters and words
- use grammatical and contextual continuity
- verify repeated names and terms
- verify the beginning and end of each paragraph
- check column transitions carefully

Do NOT invent unreadable text.

However, do NOT omit readable text merely because it is small.

If a character or word is genuinely unreadable,
preserve as much of the visible text as possible rather than
inventing a replacement.

============================================================
PART H — ACCURACY PRIORITY
============================================================

Accuracy priority is:

1. COMPLETE article coverage
2. Correct newspaper reading order
3. Correct Punjabi/Gurmukhi wording
4. Correct word and sentence boundaries
5. Correct names, places, numbers and dates
6. Correct paragraph structure
7. High fidelity to the printed article

The goal is TRANSCRIPTION, not summarization.

Prefer exact visible newspaper wording over a more natural
or more grammatically polished rewrite.

Do not "improve" the journalist's wording.

Do not correct content based on outside knowledge.

============================================================
PART I — DO NOT MIX NEIGHBORING STORIES
============================================================

Only extract text belonging to the verified target article.

Do NOT accidentally include:

- neighboring article text
- unrelated headlines
- advertisements
- unrelated captions
- page furniture
- unrelated sidebars
- unrelated boxes
- text belonging to another verified crop

When two stories are visually close, use article structure,
column boundaries, separators, headline ownership, spacing,
and paragraph continuity to keep them separate.

============================================================
PART J — STRUCTURED FIELDS
============================================================

For every article return:

headline
subheadline
author
location
date
article_text
summary
category
entities
topics
keywords
sentiment
quality

The structured fields must be based ONLY on the target article crop.

============================================================
PART K — SUMMARY
============================================================

summary is separate from article_text.

summary may be concise.

DO NOT replace article_text with the summary.

article_text must remain the COMPLETE readable Punjabi article.

============================================================
PART L — QUALITY
============================================================

quality.text_readability:

high
medium
low

quality.missing_text:

true ONLY when meaningful article content is genuinely unreadable
or physically missing from the supplied verified crop.

Do NOT mark missing_text=true simply because the article is dense,
small, difficult, or lengthy.

If the majority of the article is readable, extract it completely.

============================================================
PART M — IMAGES
============================================================

Inspect the crop for meaningful visual content belonging to this article.

Meaningful visual content includes:

- photographs
- illustrations
- maps
- charts
- graphs
- diagrams
- article-specific visual figures

Do NOT treat ordinary text as an image.

Do NOT treat the headline as an image.

Do NOT treat a caption alone as an image.

Do NOT treat decorative lines, borders, separators, bullets,
or newspaper UI elements as images.

For each actual article image return:

- image_id
- image_description
- image_bbox
- caption
- confidence

image_bbox must be measured INSIDE the supplied article crop.

Use normalized coordinates from 0 to 1000.

============================================================
PART N — CONTENT TYPE
============================================================

content_type MUST be exactly one of:

article
photo_caption
reference
advertisement
other

A genuine newspaper story with substantial independent prose
should be classified as:

article

============================================================
PART O — CONTINUATIONS
============================================================

Detect explicit continuation markers such as:

More on Page 3
Continued on Page 5
Continued from Page 1
See Page 6
Turn to Page 4
Full report on Page 7
Report on Page 2
To be continued

Resolve continuation relationships only when strong evidence exists.

Do NOT guess continuation targets.

Use headline, article subject, entities, people, organizations,
locations, events, topics, keywords, visible text, context,
and continuation markers.

Do not use image similarity alone.

Only create a continuation link when confidence >= 0.75.

============================================================
PART P — FINAL SELF-CHECK BEFORE OUTPUT
============================================================

Before returning the JSON, verify EACH ARTICLE:

1. Did I read the entire crop?
2. Did I extract every readable Punjabi paragraph?
3. Did I preserve Gurmukhi script?
4. Did I avoid translating into English?
5. Did I avoid transliteration?
6. Did I read columns in correct newspaper order?
7. Did I accidentally include text from a neighboring article?
8. Did I miss small text near the bottom or beside images?
9. Did I preserve names, places, numbers and dates?
10. Did I keep article_text as full transcription rather than summary?
11. Did I avoid inventing unreadable words?
12. Did I preserve the actual newspaper wording?

If any meaningful readable article text was omitted,
go back and re-read the crop before producing the final JSON.

============================================================
PART Q — OUTPUT
============================================================

Return ONLY valid JSON.

The JSON structure must contain:

{
    "articles": [...],
    "continuation_links": [...],
    "pending_continuations": [...]
}

Every article object MUST contain:

page
article_id
headline
subheadline
author
location
date
article_text
summary
category
entities
topics
keywords
sentiment
content_type
continuation
images
quality

============================================================
FINAL RULES
============================================================

COMPLETE TRANSCRIPTION IS MORE IMPORTANT THAN BREVITY.

Read the newspaper like a human editor reading the full story.

Extract ALL readable Punjabi article text.

Keep Punjabi in Gurmukhi script.

Never translate Punjabi into English.

Never transliterate Punjabi.

Never summarize article_text.

Never omit readable paragraphs.

Never invent unreadable content.

Never mix neighboring articles.

Never change the verified crop boundary.

Never invent article IDs.

Never create additional articles.

Return JSON only.
""".strip()