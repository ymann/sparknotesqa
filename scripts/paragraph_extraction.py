import torch, csv, json, argparse
from spacy.lang.en import English
from models import InferSent
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def get_cosine_similarity(a1, a2):
    return cosine_similarity(a1.reshape(1, -1), a2.reshape(1, -1))

def get_bert_embedding(model, summary):
    to_return = []
    for i, paragraph in enumerate(summary):
        to_return.append([])
        for sentence in paragraph:
            marked_text = "[CLS] " + sentence + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
            token_embeddings = torch.stack(encoded_layers, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings[8:, :, :] # Take mean over last 4 embeddings
            token_embeddings = torch.sum(token_embeddings, dim=0)
            token_embeddings = torch.mean(token_embeddings, dim=0)
            to_return[i].append(token_embeddings.numpy()) # convert sentence embedding form Torch tensor to numpy array
    return to_return

def process_section_pool(embedding_method, final_lst, example_id, embedded_qa, embedded_summary, summary, questions, sum_bool):
    for j, e_qa in enumerate(embedded_qa):
        best_score_a = -1000000
        best_paragraph = 1000000
        for i, e_paragraph in enumerate(embedded_summary):
            if sum_bool:
                e_paragraph = sum(e_paragraph)
            else:
                length = len(e_paragraph)
                e_paragraph = sum(e_paragraph) / length
            sim = get_cosine_similarity(e_qa.reshape(1, -1), e_paragraph.reshape(1, -1))
            if sim > best_score_a:
                best_score_a = sim
                best_paragraph = i
        answers = questions[j]["answers"]
        row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
        row.extend(answers)
        row.append(questions[j]["label"])
        final_lst.append(row)
        example_id = example_id + 1
    return final_lst, example_id

def process_section_best_sentence(embedding_method, final_lst, example_id, embedded_qa, embedded_summary, summary, questions):
    for j, e_qa in enumerate(embedded_qa):
        best_score_a = -1000000
        best_paragraph = 1000000
        for i, e_paragraph in enumerate(embedded_summary):
            best_score_p = -100000
            for e_sentence in e_paragraph:
                sim = get_cosine_similarity(e_qa.reshape(1, -1), e_sentence.reshape(1, -1))
                if sim > best_score_p:
                    best_score_p = sim
            print(best_score_p)
            if best_score_p > best_score_a:
                best_score_a = best_score_p
                best_paragraph = i
        answers = questions[j]["answers"]
        row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
        row.extend(answers)
        row.append(questions[j]["label"])
        final_lst.append(row)
        example_id = example_id + 1
    return final_lst, example_id

def process_tfidf(vectorizer, use_answers, final_lst, example_id, summary, questions):
    for j, question in enumerate(questions):
        best_score_q = -1000000
        best_paragraph = 1000000
        for i, paragraph in enumerate(summary):
            embedded_summary, embedded_question = vectorizer.fit_transform([" ".join(paragraph), question])
            embedded_summary = embedded_summary.todense()
            embedded_question = embedded_question.todense()
            sim = get_cosine_similarity(embedded_question.reshape(1, -1), embedded_summary.reshape(1, -1))
            if sim > best_score_q:
                best_score_q = sim
                best_paragraph = i
        answers = questions[j]["answers"]
        if j < len(questions) and best_paragraph < len(summary):
            row = [example_id, questions[j]["question"], " ".join(summary[best_paragraph])]
        else:
            print(str(i) + " " + str(j))
        row.extend(answers)
        row.append(questions[j]["label"])
        final_lst.append(row)
        example_id = example_id + 1

def get_context_chunks(nlp, summary, window_size):
    sentencized_chunks = []
    summary_text = " ".join([" ".join(i) for i in summary])
    if window_size == 0:
        doc = nlp(summary_text)
        doc = [[sentence] for sentence in doc]
    else:
        summary_tokens_pre = summary_text.split()
        summary_tokens = [summary_tokens_pre[x:x+window_size] for x in range(0, len(summary_tokens_pre), window_size)]
    for par in summary_tokens:
        p = " ".join(par)
        doc = nlp(p)
        chunks = [sent.string.strip() for sent in doc.sents]
        sentencized_chunks.append(chunks)
    return sentencized_chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-embedding_method", "--embedding_method", help="Embedding Method")
    parser.add_argument("-comparison_method", "--comparison_method" help="Comparison Method")
    parser.add_argument("-pool_method", "--pool_method" help="Pool Method")
    parser.add_argument("-context_size", "--context_size", type=int, help="Context Size")
    args = parser.parse_args()
    embedding_method = args.embedding_method
    comparison_method = args.comparison_method
    context_size = args.context_size
    pool_method = args.pool_method

    if embedding_method == "infersent":
        MODEL_PATH = "encoder/infersent2.pkl"
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = 'fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)
        model.build_vocab_k_words(K=100000)
    elif embedding_method == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
    elif embedding_method == "sentence_bert":
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    elif embedding_method == "tfidf":
        vectorizer = TfidfVectorizer()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    with open('data/sparknotes_dataset.json', 'r') as f:
        book_doc = json.load(f)

    final_lst = []
    example_id = 0
    counter = 0
    for doc in book_doc["data"]:
        counter = counter + 1
        if counter % 100 == 0:
            print(counter)
            print("===========================")
        questions = doc["qa_list"]
        if comparison_method == "question":
            question_list = [([qa["question"] for qa in questions if len(qa["answers"]) == 4])]
        elif comparison_method == "correct_answer":
            question_list = [[qa['question'] + '_' + qa['answers'][qa['label']] for qa in questions if len(qa["answers"]) == 4]]
        elif comparison_method == "all_answers":
            question_list = [qa['question'] + '_' + qa['answers'][0] + '_' + qa['answers'][1]
            + '_' + qa['answers'][2] + '_' + qa['answers'][3] for qa in questions if len(qa["answers"]) == 4]

        summary = doc["summary"]
        if context_size > 0:
            summary = get_context_chunks(nlp, summary, context_size)

        if len(summary) == 0 or len(question_list) == 0 or [] in question_list or [] in summary:
            continue

        if embedding_method == "tfidf":
            final_lst, example_id = process_tfidf(vectorizer, use_answers, final_lst, example_id, summary, question_list[0])

        else:
            if embedding_method == "infersent":
                embedded_summary = [model.encode(i, bsize=128, tokenize=False, verbose=False) for i in summary]
                embedded_questions = [model.encode(j, bsize=128, tokenize=False, verbose=False) for j in question_list]

            elif embedding_method == "bert":
                embedded_summary = get_bert_embedding(model, summary)
                embedded_questions = get_bert_embedding(model, question_list)

            elif embedding_method == "sentence_bert":
                embedded_summary = [model.encode(i) for i in summary]
                embedded_questions = [model.encode(i) for i in question_list]

            if pool_method == "best_sentence":
                final_lst, example_id = process_section_best_sentence(
                    embedding_method, final_lst, example_id, embedded_questions[0], embedded_summary, summary, questions)
            elif pool_method == "sum":
                final_lst, example_id = process_section_pool(embedding_method, final_lst, example_id, embedded_questions[0],
                    embedded_summary, summary, questions, True)
            elif pool_method == "average":
                final_lst, example_id = process_section_pool(embedding_method, final_lst, example_id, embedded_questions[0],
                    embedded_summary, summary, questions, False)

    output_file = 'data/paragraph_extracted_data/' + embedding_method + '_' + comparison_method + '_' + context_size + '_processed_data.csv'

    with open(output_file, 'w+') as file:
        writer = csv.writer(file)
        writer.writerows(final_lst)
