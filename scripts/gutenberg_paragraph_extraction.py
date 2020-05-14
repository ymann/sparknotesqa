# import torch, csv, json
# from models import InferSent
# from pytorch_pretrained_bert import BertTokenizer, BertModel
# from sklearn.feature_extraction.text import TfidfVectorizer
import csv, json
from spacy.lang.en import English
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def get_cosine_similarity(a1, a2):
    return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))

if __name__ == '__main__':
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    final_lst = []
    example_id = 0
    counter = 0
    with open('data/gutenberg_dataset.json', 'r') as f:
        book_doc = json.load(f)['data']

    for book in book_doc:
        if counter % 100 == 0:
            print(counter)
            print("===========================")
        sentencized_chunks = []
        book_text = book['book_text']
        book_tokens_pre = book_text.split()
        book_tokens = [book_tokens_pre[x:x+80] for x in range(0, len(book_tokens_pre), 80)]
        for par in book_tokens:
            p = " ".join(par)
            doc = nlp(p)
            chunks = [sent.string.strip() for sent in doc.sents]
            sentencized_chunks.append(chunks)

        for qa_section in book['qa_list']: # List of questions for specific chapter
            questions = qa_section['qa_list']
            question_list = [qa["question"] for qa in questions if len(qa["answers"]) == 4]
            answer_list = [qa['answers'][qa['label']] for qa in questions if len(qa["answers"]) == 4]

            embedded_book = [model.encode(i) for i in sentencized_chunks]
            embedded_questions = model.encode(question_list)

            for j, e_question in enumerate(embedded_questions):
                best_score_q = -1000000
                best_paragraph = 1000000
                for i, e_paragraph in enumerate(embedded_book):
                    best_score_p = -100000
                    for idx, e_sentence in enumerate(e_paragraph):
                        sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
                        if sim > best_score_p:
                            best_score_p = sim
                    if best_score_p > best_score_q:
                        best_score_q = best_score_p
                        best_paragraph = i
                answers = questions[j]["answers"]
                if j < len(questions) and best_paragraph < len(sentencized_chunks):
                    row = [example_id, questions[j]["question"], " ".join(sentencized_chunks[best_paragraph])]
                else:
                    print(str(i) + " " + str(j))
                    print(questions)
                    print(embedded_questions)
                    print(len(questions))
                    print(len(embedded_questions[0]))
                row.extend(answers)
                row.append(questions[j]["label"])
                final_lst.append(row)
                example_id = example_id + 1

    output_file = 'data/paragraph_extracted_data/gutenberg_sentence_bert_processed_data.csv'
    with open(output_file, 'w+') as file:
        writer = csv.writer(file)
        writer.writerows(final_lst)
