import csv, json, jsonlines

if __name__ == '__main__':
    get_short = True
    with open('data/sparknotes_data_nq_full_pred.json', 'r') as f:
        predictions = json.load(f)

    id_to_tokens = {}

    with jsonlines.open('data/sparknotes_data_nqformat.jsonl', mode='r') as reader:
        for obj in reader:
            id_to_tokens.update({ obj["example_id"]: { "example_id": obj["example_id"], "question_text": obj["question_text"],
                "document_tokens": obj["document_tokens"], "document_html": obj["document_html"] } })

    book_doc_helper = {}

    with open('sparknotes_dataset.json', 'r') as f:
        book_doc = json.load(f)

    for chapter in book_doc["data"]:
        qa_list = chapter["qa_list"]
        for qa in qa_list:
            book_doc_helper.update({ qa["question"]: { "answers": qa["answers"], "label": qa["label"] } })

    final_lst = []
    counter = 0
    no_answer_count = 0
    for p in predictions["predictions"]:
        if(counter % 500 == 0):
            print(counter)
        example_id = p["example_id"]
        doc = id_to_tokens[example_id]
        if get_short:
            start_token = p["short_answers"][0]["start_token"]
            end_token = p["short_answers"][0]["end_token"]
        else:
            start_token = p["long_answer"]["start_token"]
            end_token = p["long_answer"]["end_token"]
        if end_token > len(doc["document_tokens"]):
            continue
        start_byte = doc["document_tokens"][start_token]["start_byte"]
        end_byte = doc["document_tokens"][end_token]["end_byte"]
        if start_token == -1: # Model couldn't find answer in any of paragraphs
            chosen_paragraph = doc["document_html"][0 : 1]
            no_answer_count += 1
        else:
            chosen_paragraph = doc["document_html"][start_byte : end_byte]
        question = doc["question_text"]
        answers = book_doc_helper[question]["answers"]
        label = book_doc_helper[question]["label"]
        row = [example_id, question, chosen_paragraph]
        row.extend(answers)
        row.append(label)
        final_lst.append(row)
        counter = counter + 1
    print("no answer count", no_answer_count)
    with open('data/nq_processed_data_short.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerows(final_lst)





