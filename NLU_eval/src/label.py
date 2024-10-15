def extract_labels(dataset, dataset_name, language):
    labels = []
    for data in dataset:

        if dataset_name == "xcopa":
            if data['label'] == 0:
                label = 'A'
            elif data['label'] == 1:
                label = 'B'

        elif dataset_name == "m-mmlu":
            label = data["answer"]
            
        elif dataset_name == "kmmlu":
            if data['answer'] == 1:
                label = 'A'
            elif data['answer'] == 2:
                label = 'B'
            elif data['answer'] == 3:
                label = 'C'
            elif data['answer'] == 4:
                label = 'D'
            

        elif dataset_name == "xstory":
            if data['answer_right_ending'] == 1:
                label = 'A'
            elif data['answer_right_ending'] == 2:
                label = 'B'
        
        elif dataset_name == "xwino":
            if data['answer'] == "1":
                label = 'A'
            elif data['answer'] == "2":
                label = 'B'

        else:
            raise Exception(dataset_name)
        
        labels.append(label)

    if language == "ar":
        ar_labels = []
        for label in labels:
            if label == "A":
                ar_label = 'أ' 
            elif label == 'B':
                ar_label = 'ب'
            elif label == 'C':
                ar_label = 'ك'
            elif label == 'D':
                ar_label = 'د'
            else:
                ar_label = label 
            ar_labels.append(ar_label)
        return ar_labels

    return labels
    