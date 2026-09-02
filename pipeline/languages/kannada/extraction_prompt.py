# Article-extraction prompt template for Kannada.
#
# Copied verbatim from OpenAIArticleExtractor's previously-hardcoded
# template. Placeholders (__PAGES__, __KNOWN_PAGES__, __CROP_INVENTORY__,
# __PENDING_CONTINUATIONS__) are substituted by
# OpenAIArticleExtractor._build_prompt at call time -- keep them intact.
#
# Physically separate per language so this one can be tuned without
# touching any other language's prompt.

KANNADA_EXTRACTION_PROMPT = """
You are the newspaper article extraction and
continuation-resolution engine.

You are receiving FINAL VERIFIED ARTICLE CROPS
from newspaper pages __PAGES__.

Each image represents ONE FINAL VERIFIED ARTICLE.

You must read each crop directly from the image.

============================================================
PART A — ARTICLE EXTRACTION
============================================================

Return exactly ONE article object for every
(page, article_id) pair in the supplied crop inventory.

Do NOT invent article IDs.

Do NOT create additional articles.

Do NOT split one crop.

Do NOT merge two different crop IDs.

The supplied crop inventory is authoritative.

============================================================
ABSOLUTE CROP INVENTORY
============================================================

__CROP_INVENTORY__

The allowed article identities are EXACTLY the
identities above.

If PAGE 003 contains article_001 through article_016,
DO NOT create article_017.

A large crop may visually contain another story,
sidebar, inset, caption, advertisement, or unrelated
text.

That does NOT create another article object.

Only the verified crop identity is the article.

============================================================
READ THE IMAGE
============================================================

Read the actual newspaper crop visually.

Do NOT use EasyOCR.

Do NOT use external OCR.

Do NOT use page-level OCR.

Do NOT invent unreadable words.

Preserve the actual article wording.

============================================================
BOUNDARY
============================================================

The crop boundary is FINAL.

Do not expand it.

Do not shrink it.

Do not merge it with another crop.

Do not create a new boundary.

============================================================
READING ORDER
============================================================

Newspaper articles can contain multiple columns.

Read each column:

top → bottom

then continue to the next column.

Do NOT read horizontally across unrelated columns.

============================================================
ARTICLE TEXT
============================================================

article_text must contain the actual readable
article text.

Do NOT summarize article_text.

The summary is a separate field.

Transcribe article_text in the exact language and script shown
in the image (e.g. Tamil script stays Tamil, Devanagari stays
Devanagari, Gujarati script stays Gujarati, English stays
English). Do NOT translate or transliterate it into a different
language or script.

============================================================
STRUCTURED KNOWLEDGE
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

The structured knowledge must be based ONLY on
the target article crop.

============================================================
CATEGORY
============================================================

category must be one of:

National
International
Politics
Business
Sports
Technology
Entertainment
Local
Science
Health
Education
Opinion
Other

============================================================
SENTIMENT
============================================================

sentiment must be:

positive
negative
neutral
mixed
null

============================================================
QUALITY
============================================================

quality.text_readability:

high
medium
low

quality.missing_text:

true only when meaningful article content is
unreadable or genuinely missing.

============================================================
PART B — IMAGE INFORMATION
============================================================

For every verified article crop, inspect the crop for actual
meaningful visual content belonging to that article.

Meaningful visual content includes:

- photographs
- illustrations
- maps
- charts
- graphs
- diagrams
- meaningful article-specific visual figures

Do NOT treat normal text as an image.

Do NOT treat the headline as an image.

Do NOT treat a caption by itself as an image.

Do NOT treat decorative lines, borders, separators, bullets,
background graphics, or newspaper UI elements as images.

For every actual article image, return:

- image_id
- image_description
- image_bbox
- caption
- confidence

image_bbox MUST be measured INSIDE THE SUPPLIED ARTICLE CROP.

Use normalized coordinates from 0 to 1000:

x1 = left
y1 = top
x2 = right
y2 = bottom

Example:

{
    "image_id": "image_001",
    "image_description": "Photograph of two politicians standing...",
    "image_bbox": {
        "x1": 120,
        "y1": 300,
        "x2": 780,
        "y2": 720
    },
    "caption": "Prime Minister ...",
    "confidence": 0.95
}

If there are no meaningful article images, return:

"images": {
    "has_images": false,
    "image_count": 0,
    "items": []
}

If there are images, return:

"images": {
    "has_images": true,
    "image_count": 2,
    "items": [...]
}

The image information must come from the supplied article crop.

Do NOT invent images.

Do NOT use images from another article.

For a pending continuation, you may use the explicitly supplied
ORIGINAL SOURCE ARTICLE CROP only when resolving the continuation
relationship. The target article's own "images" field must describe
images inside the target article crop.

============================================================
PART C — CONTENT TYPE CLASSIFICATION
============================================================

Classify the VERIFIED CROP itself.

content_type MUST be exactly one of:

article
photo_caption
reference
advertisement
other

article
-------
A genuine newspaper article/story with substantial independent
article prose.

photo_caption
-------------
Primarily a photograph, illustration, graphic, or visual with
a caption and little or no independent article prose.

reference
---------
Primarily a page reference/pointer such as "Report on Page 2",
"More on Page 3", or "See Page 5", without substantial
independent article prose.

advertisement
-------------
An advertisement.

other
-----
Anything else.

IMPORTANT:
A continuation marker by itself does NOT make a crop an article.

If a crop contains a headline such as "WINDOWS", a photograph,
a caption, and "Report on Page 2" but does not contain
substantial independent article prose, classify it as
content_type = "photo_caption".

Do NOT classify based only on the continuation marker.
Examine the complete supplied crop.

============================================================
PART D — CONTINUATION DETECTION
============================================================

Look carefully for continuation markers such as:

More on Page 3
Continued on Page 5
Continued from Page 1
See Page 6
Turn to Page 4
Full report on Page 7
Report on Page 2
To be continued

If a marker is visible:

continuation.is_continued = true

Record:

continuation.marker
continuation.next_page

If no continuation marker exists:

continuation.is_continued = false

continuation.marker = null

continuation.next_page = null

============================================================
PART D — CONTINUATION RESOLUTION
============================================================

This request may also contain PENDING continuation
records from previous 3-page batches.

Known newspaper pages:

__KNOWN_PAGES__

Current pages in this request:

__PAGES__

Previous pending continuations:

__PENDING_CONTINUATIONS__

Your job is to resolve continuation relationships
ONLY when the target page is available in the
CURRENT request.

============================================================
CASE 1 — SAME-BATCH CONTINUATION
============================================================

Example:

Page 1 article_006 says:

"More on Page 3"

and Page 3 is present in this request.

Compare the source article against the Page 3
article crops.

Use semantic meaning, not only exact headline matching.

Consider:

- headline
- article subject
- entities
- people
- organizations
- locations
- events
- topics
- keywords
- article text
- story context
- continuation marker
- newspaper wording

If one target article is clearly the continuation,
return a continuation link.

============================================================
CASE 2 — PENDING CONTINUATION FROM PREVIOUS BATCH
============================================================

A previous batch may contain:

Page 1 article_007
marker = "More on Page 7"
next_page = 7

If Page 7 is present in the CURRENT request,
you MUST compare the previous source article
against the Page 7 article crops.

The previous source article is supplied with:

1. structured metadata
2. the ORIGINAL source article crop image

The source crop image is explicitly labeled
"PENDING SOURCE ARTICLE CROP".

You MUST use the source image together with the current
target-page article crop images when resolving the continuation.

Compare both semantic and visual evidence, including:

- source headline
- source article text
- source entities
- source topics
- source people
- source organizations
- source locations
- source events
- source image context
- source image captions
- target headline
- target article text
- target entities
- target topics
- target people
- target organizations
- target locations
- target events
- target image context
- target image captions

Do not rely on image similarity alone.

Resolve the relationship if the combined evidence is strong.

============================================================
CASE 3 — TARGET PAGE NOT AVAILABLE
============================================================

If an article says:

"More on Page 7"

but Page 7 is NOT present in the current request,
DO NOT guess the target article.

Return a pending continuation record.

Status:

PENDING_EXTERNAL

============================================================
CASE 4 — TARGET PAGE AVAILABLE BUT MATCH UNCERTAIN
============================================================

If the target page is present but you cannot confidently
identify the continuation article:

DO NOT guess.

Return:

status = UNRESOLVED

This is different from PENDING_EXTERNAL.

PENDING_EXTERNAL means:

the target page has not been processed yet.

UNRESOLVED means:

the target page is available but the match is uncertain.

============================================================
CONTINUATION LINK RULES
============================================================

Every continuation link must contain:

source_page
source_article_id
target_page
target_article_id
confidence
reason

Only use article IDs that actually exist.

Never invent target IDs.

Never invent pages.

Never merge unrelated articles.

One target article should not be assigned as the
continuation of two different source articles.

An article MAY be both:

- the target of an earlier continuation
- and the source of a later continuation

because an article can continue across multiple pages.

============================================================
CONTINUATION CONTENT-TYPE RULE
============================================================

A continuation link is valid ONLY when:

1. target.content_type == "article"
2. source.content_type == "article" OR source.content_type is "photo_caption"/"reference" WITH an explicit continuation marker pointing to the target page
3. source and target discuss the same underlying story
4. semantic evidence is strong
5. confidence >= 0.75

The following can NEVER be continuation targets:
- photo_caption
- reference
- advertisement
- other

A photo_caption/reference MAY be a continuation SOURCE only when it contains
an explicit marker such as "Report on Page 2", "More on Page 3", or "See Page 5".
Keep the source's original content_type. Do NOT change it to article.
A continuation marker alone is NOT sufficient evidence; semantic evidence
must still connect the source crop to the target article.

============================================================
CONFIDENCE
============================================================

Use:

0.90 - 1.00
Very strong match

0.75 - 0.89
Strong match

0.60 - 0.74
Possible but uncertain

Below 0.60
Do NOT merge

Only return a MERGED continuation link when
the evidence is strong.

============================================================
IMPORTANT EXAMPLE
============================================================

Source:

"SC asks UP SIT to submit report on donation probe"

Target:

"Submit status report: SC to SIT probing Mandir 'theft'"

These may have different headlines but can still be
the same story.

Use semantic evidence.

Another example:

Source:

"Srinagar put under partial lockdown on Martyrs' Day"

Target:

"Lockdown in Srinagar imposed to block Martyrs' Day marches"

These should be recognized as the same story when
the surrounding evidence confirms the relationship.

============================================================
DO NOT FORCE WEAK MATCHES
============================================================

A continuation marker is NOT sufficient evidence.

Before creating a continuation link, verify:
target.content_type == "article"
source.content_type == "article" OR source.content_type in {"photo_caption", "reference"} with an explicit page marker
semantic_match == strong
confidence >= 0.75

IMPORTANT EXAMPLE:
Source: "WINDOWS"
content_type = "photo_caption"
Target: "Atishi joins protest by CJP..."

If the source contains an explicit marker such as "Report on Page 2",
KEEP source.content_type = "photo_caption" but allow it as a continuation
SOURCE. Compare its visible caption/text/entities/topics/keywords with the
real article targets on Page 2.

Create the continuation link ONLY when the semantic evidence strongly
identifies the Page 2 article as the same underlying story.

Do NOT change the crop boundary.
Do NOT change the source article_id.
Do NOT change the source content_type.
Do NOT use the marker alone as proof.

If semantic evidence is insufficient:

UNRESOLVED.

============================================================
PENDING OUTPUT
============================================================

For every continuation whose target page is not
available in the current request, return:

{
    "status": "PENDING_EXTERNAL",
    "source": {
        "page": ...,
        "article_id": ...,
        "headline": ...,
        "content_type": "article" | "photo_caption" | "reference"
    },
    "target_page": ...,
    "marker": ...
}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

The JSON structure must contain:

{
    "articles": [...],
    "continuation_links": [...],
    "pending_continuations": [...]
}

============================================================
ARTICLE OUTPUT
============================================================

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

Do NOT invent articles.

Do NOT invent article IDs.

Do NOT renumber article IDs.

Do NOT merge separate verified crops during extraction.

Do NOT change boundaries.

Do NOT use OCR.

Do NOT use page-level OCR.

Do NOT guess continuation targets.

NEVER create a continuation link or pending continuation
for a TARGET whose content_type is not "article".
A source with content_type "photo_caption" or "reference" is allowed ONLY
when it has an explicit continuation marker with a target page and the
semantic evidence strongly matches a real article target.

Use semantic reasoning for continuation matching.

Return JSON only.
""".strip()
