import re


def post_process_results(dataset, dataset_name, answers, language):
    choices = []
    for data, answer in zip(dataset, answers):

        if dataset_name == "xcopa":
            if language == "ar":
                pattern = r"[（(](أ|ب)[）)]"
            else:
                pattern = r"[（(](A|B)[）)]"
            matches = re.findall(pattern, answer)
            if matches:
                answer = f"({matches[0]})"[1]
            else:
                if data['choice1'] in answer:
                    answer = 'A'
                elif data['choice2'] in answer:
                    answer = 'B'
                else:
                    answer = 'err'


        elif dataset_name == "m-mmlu":

            if language == "ar":
                pattern = r"[（(](د|ك|أ|ب)[）)]"
            else:
                pattern = r"[（(](A|B|C|D)[）)]"
            matches = re.findall(pattern, answer)
            if matches:
                answer = f"({matches[0]})"[1]
            else:
                if data['option_a'] in answer:
                    answer = 'A'
                elif data['option_b'] in answer:
                    answer = 'B'
                elif data['option_c'] in answer:
                    answer = 'C'
                elif data['option_d'] in answer:
                    answer = 'D'
                else:
                    answer = 'err'

        elif dataset_name == "xstory":
            if language == "ar":
                pattern = r"[（(](أ|ب)[）)]"
            else:
                pattern = r"[（(](A|B)[）)]"
            matches = re.findall(pattern, answer)
            if matches:
                answer = f"({matches[0]})"[1]
            else:
                if data['sentence_quiz1'] in answer:
                    answer = 'A'
                elif data['sentence_quiz2'] in answer:
                    answer = 'B'
                else:
                    answer = 'err'

        
        elif dataset_name == "xwino":
            if language == "ar":
                pattern = r"[（(](أ|ب)[）)]"
            else:
                pattern = r"[（(](A|B)[）)]"
            matches = re.findall(pattern, answer)
            if matches:
                answer = f"({matches[0]})"[1]
            else:
                if data['option1'] in answer:
                    answer = 'A'
                elif data['option2'] in answer:
                    answer = 'B'
                else:
                    answer = 'err'


        choices.append(answer)

    if language == "ar":
        ar_choices = []
        for choice in choices:
            if choice == "A":
                ar_choice = 'أ' 
            elif choice == 'B':
                ar_choice = 'ب'
            elif choice == 'C':
                ar_choice = 'ك'
            elif choice == 'D':
                ar_choice = 'د'
            else:
                ar_choice = choice 
            ar_choices.append(ar_choice)
        return ar_choices
    
    return choices
