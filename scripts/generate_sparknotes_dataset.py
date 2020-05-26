import codecs, json, argparse
from bs4 import BeautifulSoup
from spacy.lang.en import English

def get_url(example_num):
    with open("%d_url" % example_num) as f:
        doc = f.readlines()
    return doc[0]


def process_longq(example_num, document_url):
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    myh3 = soup.findAll("h3", {"class": "heading__fiveKeyQs__question"})
    question = myh3[0].text.strip()
    # question_tokens, _ = tokenize(question)
    myp = soup.findAll("p")
    answer = myp[0].text
    # tokens, _ = tokenize(answer)

    # document_tokens = [{"token": tokens[i], "start_byte": 0, "end_byte": 0, "html_token": false} for i in range(len(tokens))]
    title_html = soup.findAll("a", {"id": "tag--interior-title-link"})[0]
    title = title_html.text.strip()
    author_html = soup.findAll("a", {"id": "interiorpage_author_link1"})[0]
    author = author_html.text.strip()
    document = {"document_url": document_url, "document_html": soup.get_text(), "question": question, "answer": answer, "author": author,
    "title": title}

    return document

def process_shortqs(example_num, document_url):
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    quiz_questions = [soup.findAll("div", {"class" : "quick-quiz-question"})][0]
    questions = []
    for q_question in quiz_questions:
        curr_answers = []
        true_answer_idx = 1000
        question = ""
        children = q_question.findChildren("li", recursive=True)
        for i, child in enumerate(children):
            if len(child['class']) == 2: # Incorrect answer doesn't contain true answer class, has len 2
                curr_answers.append(child.text.strip())
            else:
                curr_answers.append(child.text.strip())
                true_answer_idx = i
        all_text = q_question.findChildren("h3", recursive=False)[0]
        question_number = all_text.findChildren("div", recursive=False)[0].text
        question = all_text.text.replace(question_number, '').strip()
        questions.append((question, curr_answers, true_answer_idx))

    title = soup.findAll("h1", {"class": "TitleHeader_title"})[0].text.strip()
    maybe_author = soup.findAll("a", {"class": "TitleHeader_authorLink"})
    if len(maybe_author) > 0: # author is link
        author = maybe_author[0].text.strip()
    else: # author is plaintext
        author = soup.findAll("div", {"class": "TitleHeader_authorName"})[0].text.strip()
    chapters = soup.findAll("h2", {"class": "interior__page__title"})[0].text.strip()
    documents = {"qa_list": [], "document_html": "", "document_url": document_url, "author": author,  "title": title,
    "chapters_covered": chapters}
    for question, answers, true_answer_idx in questions:
        document = {"question": question, "answers": answers, "label": true_answer_idx}
        documents["qa_list"].append(document)
    return documents

def process_summary(example_num, document_url, keep_analysis):
    f = codecs.open("%d_content" % example_num, 'r', 'utf-8')
    soup = BeautifulSoup(f.read(), 'html.parser')
    summary_container = soup.findAll("div", {"class": "studyGuideText"})[0]
    summary_elements = summary_container.findChildren(["h3", "h4", "p"])
    paragraphs = []
    for i, ele in enumerate(summary_elements):
        if(ele.name == "p"):
            paragraphs.append(ele.text)
        elif("Summary" in ele.text) or ("Commentary" in ele.text and i == 0 and document_url.split("/")[6] == ""): # Catch sparknotes typo where all of content is under analysis header
            continue
        elif("Analysis" in ele.text) or ("Commentary" in ele.text) or ("analysis" in ele.text) or ("Analyis" in ele.text): # Catching all of Sparknotes stupid edge cases
            if not keep_analysis:
                paragraphs.append("ANALYSIS_REACHED")
                break
        else: # subtitle - ignore
            continue
    sentencized_paragraphs = []
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for p in paragraphs:
        doc = nlp(p)
        sentences = [sent.string.strip() for sent in doc.sents]
        sentencized_paragraphs.append(sentences)

    title = soup.findAll("h1", {"class": "TitleHeader_title"})[0].text.strip()
    author = ""
    maybe_author = soup.findAll("a", {"class": "TitleHeader_authorLink"})
    if len(maybe_author) > 0: # author is link
        author = maybe_author[0].text.strip()
    else: # author is plaintext
        author = soup.findAll("div", {"class": "TitleHeader_authorName"})[0].text.strip()
    chapters = soup.findAll("h2", {"class": "interior__page__title"})[0].text.strip()
    documents = {"summary": sentencized_paragraphs, "document_html": soup.get_text(), "document_url": document_url, "author": author,  "title": title,
    "chapters_covered": chapters}
    return documents

def get_section_mappings(range_start, range_finish):
    summaries = {}
    short_qs = {}
    for i in range(range_start, range_finish):
        if i == 10930: # Broken link
            continue
        document_url = get_url(i)
        split_doc = document_url.split("/")
        book = split_doc[4]
        section = split_doc[5]
        if section == "summary":
            continue
        if len(split_doc) < 7: # ignore
            continue
        elif split_doc[6] == "": # section 1 of summary
            if book not in summaries:
                summaries[book] = {}
            if section not in summaries[book]:
                summaries[book][section] = [i]
            else:
                summaries[book][section].insert(0, i)
        elif split_doc[6][1:10] == "quickquiz":
            if book not in short_qs:
                short_qs[book] = {}
            short_qs[book][section] = i
        elif len(split_doc) == 9 and split_doc[5] != "quotes": # later section
            if book not in summaries:
                summaries[book] = {}
            if section not in summaries[book]:
                summaries[book][section] = [i]
            else:
                summaries[book][section].insert(int(split_doc[7]), i) # If index doesnt exist Python appends to end
    return summaries, short_qs

def combine_summaries(indices, keep_analysis):
    combined_summary = []
    for idx in indices:
        sum_doc = process_summary(idx, get_url(idx), keep_analysis)
        summary = sum_doc["summary"]
        for p in summary:
            if len(p) == 0: # empty paragraph
                continue
            elif p[0] == "ANALYSIS_REACHED":
                sum_doc["summary"] = combined_summary
                return sum_doc
            else:
                combined_summary.append(p)
    sum_doc["summary"] = combined_summary
    return sum_doc

def create_dataset(summaries, short_qs, keep_analysis):
    book_doc = []
    counter = 0
    for book, section_dict in summaries.items():
        if book in short_qs:
            for section, indices in section_dict.items():
                if section in short_qs[book]:
                    if counter % 500 == 0:
                        print(counter)
                    sum_doc1 = combine_summaries(indices, keep_analysis)
                    if len(sum_doc1["summary"]) == 0: # No summary
                        print(sum_doc1)
                        counter = counter + 1
                        continue
                    else:
                        curr_dict = {}
                        curr_dict["author"] = sum_doc1["author"]
                        curr_dict["title"] = sum_doc1["title"]
                        curr_dict["chapters_covered"] = sum_doc1["chapters_covered"]
                        curr_dict["summary_html"] = sum_doc1["document_html"]
                        curr_dict["summary"] = sum_doc1["summary"]
                        curr_dict["summary_url"] = sum_doc1["document_url"]
                        qa_dict = process_shortqs(short_qs[book][section], get_url(short_qs[book][section]))
                        curr_dict["qa_html"] = qa_dict["document_html"]
                        curr_dict["qa_list"] = qa_dict["qa_list"]
                        curr_dict["qa_url"] = qa_dict["document_url"]
                        book_doc.append(curr_dict)
                        counter = counter + 1
    return book_doc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-keep_analysis", "--keep_analysis", type=bool, help="Embedding Method")
    summaries, short_qs = get_section_mappings(1, 14425)
    args = parser.parse_args()
    book_doc = create_dataset(summaries, short_qs, args.keep_analysis)

    with open('data/sparknotes_dataset.json', 'w+') as file:
        json.dump({"data" : book_doc}, file)

