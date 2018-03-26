import json
import requests
from functools import reduce


def call_vision_api(image_filename, api_keys):
    api_key = api_keys['microsoft']
    post_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/ocr"
    headers = {'Ocp-Apim-Subscription-Key': api_key,
               'Content-Type': 'application/octet-stream'}
    image_data = open(image_filename, 'rb').read()
    result = requests.post(post_url, data=image_data, headers=headers)
    result.raise_for_status()

    return json.loads(result.text)

# Return a dictionary of features to their scored values (represented as lists of tuples).
# Scored values must be sorted in descending order.
#
# {
#    'feature_1' : [(element, score), ...],
#    'feature_2' : ...
# }
#
# E.g.,
#
# {
#    'tags' : [('throne', 0.95), ('swords', 0.84)],
#    'description' : [('A throne made of pointy objects', 0.73)]
# }
#


def get_standardized_result(api_result):
    output = {
        'tags': [],
        #        'categories' : [],
        #        'adult' : [],
        #        'image_types' : []
        #        'tags_without_score' : {}
    }

    for tag_data in api_result['regions']:
        lines = list(map(
            lambda x: x['words'],
            tag_data['lines']
        ))
        lines = reduce(lambda x, y: x + y, lines)

        words = list(map(lambda x: x['text'], lines))

        for word in words:
            output['tags'].append((word, 0.0))

    # for caption in api_result['description']['captions']:
    #     output['captions'].append((caption['text'], caption['confidence']))

#    for category in api_result['categories']:
#        output['categories'].append(([category['name'], category['score']))

#    output['adult'] = api_result['adult']

#    for tag in api_result['description']['tags']:
#        output['tags_without_score'][tag] = 'n/a'

#    output['image_types'] = api_result['imageType']

    return output
