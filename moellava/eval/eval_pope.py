import os
import json
import argparse


def normalize_question_text(text):
    """Normalize question text to handle typos like 'imange' -> 'image'."""
    return text.replace('imange', 'image')


def eval_pope(pred_list, label_list):
    """
    Evaluate POPE given aligned prediction and label lists.

    Args:
        pred_list: List of 0/1 (model predictions)
        label_list: List of 0/1 (ground truth)
    """
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

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))


def parse_answer(text):
    """Convert model output text to binary prediction (0=no, 1=yes)."""
    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]
    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        return 0
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    # Load questions: map question_id -> {image, category, text}
    questions = {}
    with open(args.question_file, 'r') as f:
        for line in f:
            q = json.loads(line)
            questions[q['question_id']] = q

    # Load model answers
    answers = [json.loads(line) for line in open(args.result_file)]

    for file in sorted(os.listdir(args.annotation_dir)):
        if not file.startswith('coco_pope_') or not file.endswith('.json'):
            continue
        category = file[10:-5]

        # Build ground truth dict keyed by (image, normalized_question_text)
        # This composite key is unique — same question text appears on different images
        # with different labels.
        label_dict = {}
        annotation_path = os.path.join(args.annotation_dir, file)
        with open(annotation_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                key = (entry['image'], normalize_question_text(entry['text']))
                label_dict[key] = 1 if entry['label'] == 'yes' else 0

        # Filter answers for this category and match by (image, question_text)
        cur_answers = [a for a in answers if questions[a['question_id']]['category'] == category]

        pred_list = []
        label_list = []
        skipped = 0

        for answer in cur_answers:
            qid = answer['question_id']
            q = questions[qid]
            image = q['image']

            # Extract question text from prompt (remove instruction suffix)
            prompt = answer.get('prompt', '')
            question_text = prompt.split('\n')[0] if '\n' in prompt else prompt
            question_text = normalize_question_text(question_text)

            key = (image, question_text)
            if key not in label_dict:
                skipped += 1
                continue

            label_list.append(label_dict[key])
            pred_list.append(parse_answer(answer['text']))

        print('Category: {}, # samples: {}, matched: {}, skipped: {}'.format(
            category, len(cur_answers), len(pred_list), skipped))
        eval_pope(pred_list, label_list)
        print("====================================")
