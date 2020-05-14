import csv
import json

def get_book_qas(title, book_doc):
    qa_list = []
    author = ""
    for chapter in book_doc:
        if chapter['title'].lower() in title.lower():
            qa = chapter['qa_list']
            chapters_covered = chapter['chapters_covered']
            qa_list.append({'chapters_covered': chapters_covered, 'qa_list': qa})
            author = chapter['author']
    return qa_list, author

def get_summed_summaries_length(title, book_doc):
    length = 0
    num_summaries = 0
    for chapter in book_doc:
        if chapter['title'].lower() in title.lower():
            summary = " ".join([" ".join(i) for i in chapter["summary"]])
            length += len(summary.split(' '))
            num_summaries += 1
    if num_summaries == 0:
        return 0
    return length / num_summaries

if __name__ == '__main__':
    rows = []
    gutenberg_doc = []
    with open('data/paragraph_extracted_data/matches_with_keys2.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    keys = [row[0] for row in rows]
    titles = [row[1] for row in rows]
    with open('data/sparknotes_dataset.json', 'r') as f:
        book_doc = json.load(f)['data']

    for idx, key in enumerate(keys):
        title = titles[idx]
        # with open('../project_gutenberg/raw/book_%s' % key) as f:
        #     book = f.read()
        #     qa_list, author = get_book_qas(title, book_doc)
        #     gutenberg_doc.append({'author': author, 'title': title, 'qa_list': qa_list, 'book_text': book})
            # num_q_per_chapter = sum([len(qa['qa_list']) for qa in qa_list])
            # for line in book:
            #     new_book.append([str(token) for token in nlp(line)])
            # num_words = [len(line) for line in new_book]
        print(get_summed_summaries_length(title, book_doc))

    # with open('semanticsearch/data/gutenberg_dataset.json', 'w+') as file:
    #     json.dump({"data" : gutenberg_doc}, file)


