"""Check gated files without downloading weights or printing credentials."""
from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from requests.exceptions import RequestException


def check_model_access():
    token = get_token()
    accessible = True
    for repo, filename in [
        ('sesame/csm-1b', 'prompts/conversational_a.wav'),
        ('meta-llama/Llama-3.2-1B', 'tokenizer.json'),
    ]:
        try:
            get_hf_file_metadata(hf_hub_url(repo, filename), token=token, timeout=15)
            print(f'Model access: {repo}: OK')
        except GatedRepoError:
            accessible = False
            print(f'Model access: {repo}: approval or token permission is missing.')
            print(f'  Check access at https://huggingface.co/{repo} using the account that owns your token.')
        except (HfHubHTTPError, RequestException) as error:
            accessible = False
            response = getattr(error, 'response', None)
            print(f'Model access: {repo}: {type(error).__name__}, HTTP {getattr(response, "status_code", None)}. Check connection and token permissions.')
    if not accessible:
        print('Generation paused. Wait for both model approvals; if needed, log in with .\\.venv\\Scripts\\huggingface-cli.exe login.')
    return accessible


if __name__ == '__main__':
    raise SystemExit(0 if check_model_access() else 1)
