import json, random, argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def get_cosine_similarity(a1, a2):
    return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))

def calculate_mmr(rankings):
    running_sum = 0
    for ranking, label in rankings:
        rank = ranking.index(label) + 1
        running_sum += 1 / rank
    return running_sum / len(rankings)


def process_section_best_sentence(embedding_method, comp_method, embedded_questions, embedded_summary):
    ranked_paragraphs = []
    if comp_method < 2:
        for e_question in embedded_questions:
            scores = []
            for i, e_paragraph in enumerate(embedded_summary):
                best_score_p = -100000
                for e_sentence in e_paragraph:
                    sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
                    if sim > best_score_p:
                        best_score_p = sim
                scores.append((i, best_score_p))
            sorted_scores = sorted(scores, key=lambda x: x[1]*-1)
            ranked_paragraphs.append([x[0] for x in sorted_scores])
    else:
        embedded_questions_pre = embedded_questions
        embedded_questions = [embedded_questions_pre[x:x+4] for x in range(0, len(embedded_questions_pre), 4)]
        for e_question_set in embedded_questions:
            scores = []
            best_score_a = -100000
            best_p_a = -100000
            for e_question in e_question_set:
                for i, e_paragraph in enumerate(embedded_summary):
                    best_score_p = -100000
                    best_p = -100000
                    for e_sentence in e_paragraph:
                        sim = get_cosine_similarity(e_question.reshape(1, -1), e_sentence.reshape(1, -1))
                        if sim > best_score_p:
                            best_score_p = sim
                            best_p = i
                    if best_score_p > best_score_a:
                        best_score_a = best_score_p
                        best_p_a = best_p
            scores.append((best_p_a, best_score_a))
            sorted_scores = sorted(scores, key=lambda x: x[1]*-1)
            ranked_paragraphs.append([x[0] for x in sorted_scores])
    return ranked_paragraphs

def process_tfidf(vectorizer, comp_method, summary, questions):
    ranked_paragraphs = []
    if comp_method < 2:
        for j, question in enumerate(questions):
            scores = []
            for i, paragraph in enumerate(summary):
                embedded_summary, embedded_question = vectorizer.fit_transform([" ".join(paragraph), question])
                embedded_summary = embedded_summary.todense()
                embedded_question = embedded_question.todense()
                sim = get_cosine_similarity(embedded_question.reshape(1, -1), embedded_summary.reshape(1, -1))
                scores.append((i, sim))
            sorted_scores = sorted(scores, key=lambda x: x[1]*-1)
            ranked_paragraphs.append([x[0] for x in sorted_scores])
    else:
        questions_pre = questions
        questions = [questions_pre[x:x+4] for x in range(0, len(questions_pre), 4)]
        for question_set in questions:
            scores = []
            best_score_a = -100000
            best_p_a = -100000
            for j, question in enumerate(question_set):
                for i, paragraph in enumerate(summary):
                    embedded_summary, embedded_question = vectorizer.fit_transform([" ".join(paragraph), question])
                    embedded_summary = embedded_summary.todense()
                    embedded_question = embedded_question.todense()
                    sim = get_cosine_similarity(embedded_question.reshape(1, -1), embedded_summary.reshape(1, -1))
                    if sim > best_score_a:
                        best_score_a = sim
                        best_p_a = i
            scores.append((best_p_a, sim))
            sorted_scores = sorted(scores, key=lambda x: x[1]*-1)
            ranked_paragraphs.append([x[0] for x in sorted_scores])
    return ranked_paragraphs

def get_random_paragraphs(length):
    ranked_paragraphs = []
    for i in range(5):
        lst = [i for i in range(length)]
        ranked_paragraphs.append(random.sample(lst, length))
    return ranked_paragraphs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-embedding_method", "--embedding_method", help="Embedding Method")
    parser.add_argument("-comparison_method", "--comparison_method", type=int, help="Comparison Method")
    parser.add_argument("-context_size", "--context_size", type=int, help="Context Size")
    args = parser.parse_args()

    embedding_method = args.embedding_method
    comp_method = args.comparison_method
    context_size = args.context_size
    pool_method = 0

    if embedding_method == "sentence_bert":
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    elif embedding_method == "tfidf":
        vectorizer = TfidfVectorizer()

    with open('data/paragraph_extraction_testset.json', 'r') as f:
        book_doc = json.load(f)

    rankings = []
    counter = 0
    for doc in book_doc["data"]:
        counter = counter + 1
        if counter % 5 == 0:
            print(counter)
            print("===========================")
        questions = doc["qa_list"]

        if comp_method == 0:
            question_list = [[qa["question"] for qa in questions if len(qa["answers"]) == 4]]
        elif comp_method == 1:
            question_list = [[qa['question'] + '_' + qa['answers'][0] + '_' + qa['answers'][1] + '_' + qa['answers'][2] + '_' +
            qa['answers'][3] for qa in questions if len(qa["answers"]) == 4]]
        else:
            question_list = [[qa['question'] + '_' + qa['answers'][i] for i in range(4) for qa in questions if len(qa["answers"]) == 4]]

        summary = doc["summary"]

        if len(summary) == 0 or len(question_list) == 0 or [] in question_list or [] in summary:
            continue

        if embedding_method == "tfidf":
            ranked_paragraphs = process_tfidf(vectorizer, comp_method == 2, summary, question_list[0])

        elif embedding_method == "sentence_bert":
            embedded_summary = [model.encode(i) for i in summary]
            embedded_questions = [model.encode(i) for i in question_list]
            ranked_paragraphs = process_section_best_sentence(embedding_method, comp_method == 2, embedded_questions[0], embedded_summary)
        else: # random guessing
            ranked_paragraphs = get_random_paragraphs(len(summary))
        p_labels = [qa['p_label'] for qa in questions if len(qa['answers']) == 4]
        rankings.extend(zip(ranked_paragraphs, p_labels))
    print("MMR Score: %d" % calculate_mmr(rankings))




