import re


class EntityExtractor:
    """
    Lightweight Named Entity Extractor.

    Version 1

    Extracts probable:

    - People
    - Organizations
    - Places
    - Dates
    """

    PERSON_PREFIXES = {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Shri",
        "Smt",
        "Sri",
    }

    MONTHS = {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }

    def extract(self, text: str):

        if not text:

            return {
                "people": [],
                "organizations": [],
                "places": [],
                "dates": [],
            }

        entities = {

            "people": [],
            "organizations": [],
            "places": [],
            "dates": [],
        }

        # -----------------------------------------
        # Capitalized phrases
        # -----------------------------------------

        phrases = re.findall(

            r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",

            text,
        )

        for phrase in phrases:

            words = phrase.split()

            if words[0] in self.MONTHS:

                entities["dates"].append(phrase)

                continue

            if len(words) >= 2:

                entities["people"].append(phrase)

        # -----------------------------------------
        # Organizations
        # -----------------------------------------

        org_keywords = {

            "Ltd",
            "Limited",
            "Bank",
            "University",
            "Corporation",
            "Ministry",
            "Government",
            "Company",
            "Inc",
        }

        for keyword in org_keywords:

            if keyword in text:

                matches = re.findall(

                    rf"[A-Z][A-Za-z& ]+{keyword}",

                    text,
                )

                entities["organizations"].extend(matches)

        # -----------------------------------------
        # Dates
        # -----------------------------------------

        date_pattern = re.findall(

            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",

            text,
        )

        entities["dates"].extend(date_pattern)

        # -----------------------------------------
        # Remove duplicates
        # -----------------------------------------

        for key in entities:

            entities[key] = sorted(
                set(entities[key])
            )

        return entities