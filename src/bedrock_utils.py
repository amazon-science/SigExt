import json
import logging
import time
import traceback

import boto3.session as boto3_session
import botocore.config


def extract_xml_tag(generation: str, tag):
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return
    value = generation[begin:end].strip()
    return value


def predict_one_eg_mistral(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": "mistral.mistral-7b-instruct-v0:2",
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {"max_tokens": 512, "temperature": 1.0, "top_p": 0.8, "top_k": 10, "prompt": x["prompt_input"]}

    api_template["body"] = json.dumps(body)

    success = False
    response = None
    for i in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(5)

    if success:
        response_body = json.loads(response.get("body").read())
        logging.info(response_body)
        return response_body["outputs"][0]["text"]
    else:
        return ""


def predict_one_eg_claude_instant(x):
    current_session = boto3_session.Session()
    bedrock = current_session.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        config=botocore.config.Config(
            read_timeout=120,  # corresponds to inference time limit set for Bedrock
            connect_timeout=120,
            retries={
                "max_attempts": 5,
            },
        ),
    )
    api_template = {
        "modelId": "anthropic.claude-instant-v1",
        "contentType": "application/json",
        "accept": "*/*",
        "body": "",
    }

    body = {
        "max_tokens_to_sample": 512,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 10,
        "prompt": "Human: {prompt}\nWrite your summary in <summary> XML tags.\n\nAssistant: ".format(
            prompt=x["prompt_input"].strip()
        ),
    }

    api_template["body"] = json.dumps(body)

    success = False
    response = None
    for i in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(20)

    if success:
        response_body = json.loads(response.get("body").read())
        summary = extract_xml_tag(response_body["completion"], "summary")
        logging.info(summary or response_body["completion"])
        return summary or response_body["completion"]
    else:
        return ""
