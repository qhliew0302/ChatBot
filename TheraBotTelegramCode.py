import pickle
import json
import random
import os.path

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes = ["neutral", "happy", "sad", "love", "anger"]

model = load_model("therabot.h5")

responses_json = json.load(open('responses.json'))

emotion_scores = {
    "neutral": 0,
    "happy": 0,
    "sad": 0,
    "love": 0,
    "anger": 0
}


def emotion_score(y_prob, user_id):
    filename = str(user_id) + '.txt'
    file_exists = os.path.exists(filename)

    if file_exists:
        f = open(filename, 'r')
        score_str = f.readlines()
        i = 0
        for emotion in emotion_scores:
            score = float(score_str[i])
            emotion_scores[emotion] = score
            # print(str(user_id) + str(emotion_scores[emotion]))
            i = i + 1
        emotion_scores["neutral"] += y_prob[0][0]
        emotion_scores["happy"] += y_prob[0][1]
        emotion_scores["sad"] += y_prob[0][2]
        emotion_scores["love"] += y_prob[0][3]
        emotion_scores["anger"] += y_prob[0][4]
        f.close()
        f = open(filename, 'w')
        for emotion in emotion_scores:
            # print(str(user_id) + str(emotion_scores[emotion]))
            f.write(str(emotion_scores[emotion]) + '\n')
        f.close()
    else:

        for emotion in emotion_scores:
            emotion_scores[emotion] = 0

        emotion_scores["neutral"] += y_prob[0][0]
        emotion_scores["happy"] += y_prob[0][1]
        emotion_scores["sad"] += y_prob[0][2]
        emotion_scores["love"] += y_prob[0][3]
        emotion_scores["anger"] += y_prob[0][4]

        f = open(filename, 'w')
        for emotion in emotion_scores:
            # print(str(user_id) + str(emotion_scores[emotion]) + " ")
            f.write(str(emotion_scores[emotion]) + '\n')
        f.close()


def get_highest_key(user_id):
    highest_key = ""
    filename = str(user_id) + '.txt'
    f = open(filename, 'r')
    highest_value = 0
    score_str = f.readlines()
    i = 0
    for emotion in emotion_scores:
        score = float(score_str[i])
        emotion_scores[emotion] = score
        # print(str(user_id) + str(emotion_scores[emotion]))
        i = i + 1

    for key in emotion_scores.keys():
        if emotion_scores[key] >= highest_value:
            highest_value = emotion_scores[key]
            highest_key = key

    f.close()
    f = open(filename, 'w')
    for j in range(5):
        f.write("0" + '\n')
    f.close()

    return highest_key


def consolidation_message(highest_key):
    if highest_key == 'neutral':
        return "You seem to be a neutral person with their feelings in check"
    elif highest_key == 'happy':
        return "You seem to be quite content with your life. I wish you stay this way!"
    elif highest_key == 'sad':
        return "You seem to be feeling a little heavy. I would recommend talking to a close friend or a therapist."
    elif highest_key == 'love':
        return "Your life seems to be filled with love, I hope you feel this way forever!"
    elif highest_key == 'anger':
        return "You sound a little cross, I would recommend doing something that makes you cal"
    else:
        return "Goodbye! Take care"


def reply(detected_intent):
    for i in range(5):
        if responses_json['intents'][i]['tag'] == detected_intent:
            # print(responses_json['intents'][i]['responses'][random.randrange(0, len(responses_json['intents'][i]['responses']))])
            return str(responses_json['intents'][i]['responses'][
                           random.randrange(0, len(responses_json['intents'][i]['responses']))])


def fallback_intent():
    return "Sorry I don't understand. Can you elaborate please?"


def analyze_message(user_message):
    text = [user_message]
    sequences_test = tokenizer.texts_to_sequences(text)
    MAX_SEQUENCE_LENGTH = 30
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH - 5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    return data_test


def predict_emotion(data_test, y_prob):
    #     y_prob = model.predict(data_test)
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
    #         print(y_prob[0])
    return pred


def responses(user_message, user_id):
    if user_message != "quit":
        data_test = analyze_message(user_message)
        y_prob = model.predict(data_test)
        pred = predict_emotion(data_test, y_prob)
        highest_emotion_confidence = y_prob[0][pred]
        emotion_score(y_prob, user_id)
        if highest_emotion_confidence > 0.33:
            # print(emotion_scores)
            return reply(classes[pred])
        else:
            # print(emotion_scores)
            return fallback_intent()
    elif user_message.lower() == "quit":
        highest_key = get_highest_key(user_id)
        return consolidation_message(highest_key)
