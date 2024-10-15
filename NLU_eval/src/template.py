def xcopa_template(data, language):
    common_en = 'You are an expert in commonsense reasoning in SPEC_LANGUAGE.' + \
    '\nHere is a premise and a question.\nPremise: ' + \
    data['premise'] + '\nQuestion: ' + data['question'] + \
    'What is it?\nA. ' + \
    data['choice1'] + \
    '\nB. ' +  \
    data['choice2'] + \
    '\nPlease choose the correct answer from the options. Only choose the answer, no need for explanation.' + \
    '\nAnswer: '
    ar = common_en.replace("SPEC_LANGUAGE", "Arabic")
    ar = ar.replace("A.", 'أ.' )
    ar = ar.replace("B.", 'ب.' )
    ar = ar.replace("C.", 'ك.' )
    ar = ar.replace("D.", 'د.' )
    dict = {
        "zh": common_en.replace("SPEC_LANGUAGE", "Chinese"),
        "jp": common_en.replace("SPEC_LANGUAGE", "Japanese"),
        "ko": common_en.replace("SPEC_LANGUAGE", "Korean"),
        "ar": ar,
        "sw": common_en.replace("SPEC_LANGUAGE", "Swahili"),
        "bn": common_en.replace("SPEC_LANGUAGE", "Bengali"),
        "en": common_en.replace("SPEC_LANGUAGE", "English")
    }
    return dict[language]



def xstory_template(data, language):
    common_en = 'You are an expert in commonsense reasoning in SPEC_LANGUAGE.' + \
        ' Here is a story and a question.\nStory: ' + \
        data['input_sentence_1'] + data['input_sentence_2'] + data['input_sentence_3'] + data['input_sentence_4'] + \
        '\nQuestion: Which of the following options is the more reasonable inference?' + \
        '\nA. ' + data['sentence_quiz1'] + \
        '\nB. ' + data['sentence_quiz2'] + \
        '\nPlease choose the correct answer from the options. Only choose the answer, no need for explanation.' + \
        '\nAnswer: '
    ar = common_en.replace("SPEC_LANGUAGE", "Arabic")
    ar = ar.replace("A.", 'أ.' )
    ar = ar.replace("B.", 'ب.' )
    ar = ar.replace("C.", 'ك.' )
    ar = ar.replace("D.", 'د.' )
    dict = {
        "zh": common_en.replace("SPEC_LANGUAGE", "Chinese"),
        "jp": common_en.replace("SPEC_LANGUAGE", "Japanese"),
        "ko": common_en.replace("SPEC_LANGUAGE", "Korean"),
        "ar": ar,
        "sw": common_en.replace("SPEC_LANGUAGE", "Swahili"),
        "bn": common_en.replace("SPEC_LANGUAGE", "Bengali"),
        "en": common_en.replace("SPEC_LANGUAGE", "English")
    }
    return dict[language]



def xwino_template(data, language):
    common_en = 'You are an expert in sentence completion in SPEC_LANGUAGE.' + \
        ' Here is an incomplete sentence and a question.\nIncomplete sentence: ' + \
        data['sentence'] + '\nQuestion: Which of the following options is the more reasonable story completion?' + \
        '\nA. ' + data['option1'] + \
        '\nB. ' + data['option2'] + \
        '\nPlease choose the correct answer from the options. Only choose the answer, no need for explanation.' + \
        '\nAnswer: '
        
    ar = common_en.replace("SPEC_LANGUAGE", "Arabic")
    ar = ar.replace("A.", 'أ.' )
    ar = ar.replace("B.", 'ب.' )
    ar = ar.replace("C.", 'ك.' )
    ar = ar.replace("D.", 'د.' )
    dict = {
        "zh": common_en.replace("SPEC_LANGUAGE", "Chinese"),
        "jp": common_en.replace("SPEC_LANGUAGE", "Japanese"),
        "ko": common_en.replace("SPEC_LANGUAGE", "Korean"),
        "ar": ar,
        "sw": common_en.replace("SPEC_LANGUAGE", "Swahili"),
        "bn": common_en.replace("SPEC_LANGUAGE", "Bengali"),
        "en": common_en.replace("SPEC_LANGUAGE", "English")
    }
    return dict[language]





def mmmlu_template(data, language):
    common_en = 'I want you to play the role of an answering expert in SPEC_LANGUAGE.' + \
        ' Here is a question.\nQuestion: ' + \
        data['instruction'] + \
        '\nA. ' +  data['option_a'] + \
        '\nB. ' +  data['option_b'] + \
        '\nC. ' +  data['option_c'] + \
        '\nD. ' +  data['option_d'] + \
        '\nPlease choose the correct answer from the options. Only choose the answer, no need for explanation.' + \
        '\nAnswer: '
    ar = common_en.replace("SPEC_LANGUAGE", "Arabic")
    ar = ar.replace("A.", 'أ.' )
    ar = ar.replace("B.", 'ب.' )
    ar = ar.replace("C.", 'ك.' )
    ar = ar.replace("D.", 'د.' )
    dict = {
        "zh": common_en.replace("SPEC_LANGUAGE", "Chinese"),
        "jp": common_en.replace("SPEC_LANGUAGE", "Japanese"),
        "ko": common_en.replace("SPEC_LANGUAGE", "Korean"),
        "ar": ar,
        "sw": common_en.replace("SPEC_LANGUAGE", "Swahili"),
        "bn": common_en.replace("SPEC_LANGUAGE", "Bengali"),
        "en": common_en.replace("SPEC_LANGUAGE", "English")
    }
    return dict[language]


def kmmlu_template(data, language):
    common_en = 'I want you to play the role of an answering expert in SPEC_LANGUAGE.' + \
        ' Here is a question.\nQuestion: ' + \
        data['question'] + \
        '\nA. ' +  data['A'] + \
        '\nB. ' +  data['B'] + \
        '\nC. ' +  data['C'] + \
        '\nD. ' +  data['D'] + \
        '\nPlease choose the correct answer from the options. Only choose the answer, no need for explanation.' + \
        '\nAnswer: '
    ar = common_en.replace("SPEC_LANGUAGE", "Arabic")
    ar = ar.replace("A.", 'أ.' )
    ar = ar.replace("B.", 'ب.' )
    ar = ar.replace("C.", 'ك.' )
    ar = ar.replace("D.", 'د.' )
    dict = {
        "zh": common_en.replace("SPEC_LANGUAGE", "Chinese"),
        "jp": common_en.replace("SPEC_LANGUAGE", "Japanese"),
        "ko": common_en.replace("SPEC_LANGUAGE", "Korean"),
        "ar": ar,
        "sw": common_en.replace("SPEC_LANGUAGE", "Swahili"),
        "bn": common_en.replace("SPEC_LANGUAGE", "Bengali"),
        "en": common_en.replace("SPEC_LANGUAGE", "English")
    }
    return dict[language]