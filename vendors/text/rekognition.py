import boto3


def call_vision_api(image_filename, api_keys):
    client = boto3.client('rekognition')

    with open(image_filename, 'rb') as image:
        return get_text_detection(client, image)


def get_text_detection(client, image):
    response = client.detect_text(Image={'Bytes': image.read()})
    return response


def get_standardized_result(api_result):
    output = {
        'tags': [],
    }
    if 'TextDetections' not in api_result:
        return output

    labels = api_result['TextDetections']
    for tag in labels:
        output['tags'].append((tag['DetectedText'], tag['Confidence'] / 100))

    return output
