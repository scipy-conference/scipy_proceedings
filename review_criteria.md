# Suggested Review Criteria for SciPy Proceedings

These review criteria were created to help guide authors and reviewers alike.
Suggestions and amendments to these review criteria are enthusiastically welcomed
via discussion or pull request.

## Written Quality

- Prose should be written in English.
- Prose should clearly and consistently communicate the narrative (e.g. consistent
  tense and person; use of active voice; etc)
- The written style should convey information that furthers the knowledge or
  research of the reader.
- Due to the interdisciplinary nature of SciPy, highly domain-specific jargon
  should be avoided or explained where possible.
- Papers that do not meet quality standards may not be approved for publication.

## Technical Content

- The technical content should be scientifically sound.
- Computational content should, likewise, be verified.
- The work should describe the development or use of python software for
  approaching a problem in a domain within the scope of the conference.

## Novelty and Impact

- Assume that the novelty of the conceptual work described in the paper has been assessed by the Program Committee.
  The paper should advance the state of a scientific domain, the practice
  of scientific computing itself, or another subject area within the scope of
  the conference.

## Verifiability

- Software descriptions should be accompanied by references to or examples of
  representative source code.
- Source code essential to the conclusions of the paper should be made
  available to the reader.
- Data sources should be identified (e.g., with citation to a persistent DOI).
- Analysis should be accompanied by a workflow description sufficient
  to reproduce the results.

## Other Requirements

- All mentioned software should be formally and consistently cited wherever
  possible.
- Acronyms should be spelled out upon first mention.
- License conditions on images and figures must be respected (Creative Commons,
  etc.).
- Mathematical and other symbols should be defined.
- Definitions should include consistent units where appropriate.
- Avoid long paragraphs of your rich text file to enable reviewers to target
  specific sections with their comments. GitHub comments are per line of text.
- Avoid custom LaTeX markup where possible.

### Length

- The compiled version should be no longer than 6000 words, including figure text,
  source code, and appendices. References are excluded from this limit.

### Figures

- All figures and tables should have captions.
- Figure text should be of a readable size.
- Images and figures should be reasonably sized and formatted for viewing online;
  typically a few hundred kilobytes and less than 1 MB.

### Code Snippets

- Code snippets should be formatted to fit inside a single column without
  overflow. Please make sure to check the build server's copy of your
  paper; it's the authoritative view of how your paper will appear.
- If not in Python, the language in the code snippet should be mentioned.
- Code snippets should follow a common style guide. PEP8 is preferred.

# Suggestions for Writing Great Feedback

* Use summary feedback for overarching themes of your feedback. In-line feedback gives
  greater context to your comments; treat the paper as a code review.
* Great feedback is respectful, direct, and actionable. You needn't provide examples of
  how to correct the paper for repeated issues (e.g. tense, narrative clarity), but
  providing suggestions for the first few incidents and then referencing them later
  will help authors while reducing reviewer time.
  * Instead of: "Sentence is too long and wordy."
  * Consider: "The entire paragraph is comprised of 2 sentences, making it hard to follow.
    Either consolidate it into the previous or next paragraph or flesh out the existing
    paragraph more."
  * Ideally (and time permitting): Provide phrasing and word fragments such as "Consider
    rewording first sentence to avoid ending with a preposition. Instead of `First, we
    identified essays that we wanted to run analysis on`, consider `First, we identified
    essays for analysis`".
* Consider using M. Pacer's [Proselint](https://github.com/amperser/Proselint) tool to help
  automate identification of some of the more common errors so you can focus more on content
  and less on style.
* Familiarize yourself with [Github's markdown guide](https://guides.github.com/features/mastering-markdown/)
  if this is your first time as a reviewer.
* If you are not a domain expert in the field of the paper, make sure to let the authors
  know and focus on the structure and narrative of the paper. Good papers should be layered
  such that someone new to the domain can follow the narrative, but an expert can grasp its
  details.

# Proceedings Committee

* Feel free to engage the [SciPy Proceedings Committee](https://github.com/scipy-conference/scipy_proceedings/blob/master/README.md#contacting-the-proceedings-co-chairs) with questions, or if you need a replacement or additional reviewer.
