import os
import json
import argparse

def normalize_question_text(text):
    """
    Normalize question text for matching.
    Handles typos like 'imange' -> 'image'.
    """
    text = text.replace('imange', 'image')  # Fix common typo
    return text

def eval_pope(answers, text_label_dict):
    """
    Evaluate POPE by matching answers to ground truth labels by question text.

    Args:
        answers: List of dicts with 'question_id', 'prompt', and 'text' (model output)
        text_label_dict: Dict mapping question_text -> 'yes'/'no' (ground truth)
    """
    pred_list = []
    label_list = []
    skipped = 0

    for answer in answers:
        # Extract question text from prompt (remove the answer instruction)
        prompt = answer.get('prompt', '')
        # Remove the instruction line "Answer the question using a single word or phrase."
        question_text = prompt.split('\n')[0] if '\n' in prompt else prompt

        # Normalize text to handle typos
        question_text = normalize_question_text(question_text)

        # Match by question text, not by question_id
        # (question_ids don't align across the different category files)
        if question_text not in text_label_dict:
            print(f'Warning: question text not found in annotation file: {question_text[:50]}...')
            skipped += 1
            continue

        gt_label = text_label_dict[question_text]
        label_list.append(1 if gt_label == 'yes' else 0)

        # Process model prediction
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            pred_list.append(0)
        else:
            pred_list.append(1)

    if skipped > 0:
        print(f'Skipped {skipped} answers due to missing question_id in annotations')

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    if TP + FP > 0:
        precision = float(TP) / float(TP + FP)
    else:
        precision = 0.0

    if TP + FN > 0:
        recall = float(TP) / float(TP + FN)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2*precision*recall / (precision + recall)
    else:
        f1 = 0.0

    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]

        # Load annotation file into a dict keyed by question TEXT (not question_id)
        # because question_ids don't align across categories in our data
        text_label_dict = {}
        annotation_path = os.path.join(args.annotation_dir, file)
        with open(annotation_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Normalize text to match question file
                normalized_text = normalize_question_text(entry['text'])
                text_label_dict[normalized_text] = entry['label']

        # Filter answers by category
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        eval_pope(cur_answers, text_label_dict)
        print("====================================")
