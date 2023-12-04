import requests
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GitHub API setup
token = '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'
headers = {'Authorization': f'token {token}'}

# Function to search repositories
def search_repos(query):
    url = f'https://api.github.com/search/repositories?q={query}'
    response = requests.get(url, headers=headers)
    return response.json()['items']

# Function to recursively list files in a repository
def list_files_in_repo(repo_full_name, path=''):
    url = f'https://api.github.com/repos/{repo_full_name}/contents/{path}'
    response = requests.get(url, headers=headers)
    files = response.json()

    file_paths = []
    for file in files:
        if file['type'] == 'file' and (file['name'].endswith('.py') or file['name'].endswith('.md')):
            file_paths.append(file['path'])
        elif file['type'] == 'dir':
            file_paths.extend(list_files_in_repo(repo_full_name, file['path']))
    
    return file_paths

# Function to get the content of a file
def get_file_content(repo, file_path):
    url = f'https://api.github.com/repos/{repo}/contents/{file_path}'
    response = requests.get(url, headers=headers)
    content = response.json().get('content', '')
    return content

# Function to save content to JSONL
def save_to_jsonl(contents, filename='output.jsonl'):
    with open(filename, 'w') as file:
        for content in contents:
            json_record = json.dumps({"text": content})
            file.write(json_record + '\n')

# Main process
repositories = search_repos('Taipy')
logging.info(f'Found {len(repositories)} repositories with "Taipy"')

file_contents = []

for repo in repositories:
    logging.info(f'Processing repository: {repo["full_name"]}')
    try:
        file_paths = list_files_in_repo(repo['full_name'])
        logging.info(f'Found {len(file_paths)} .py and .md files in {repo["full_name"]}')

        for file_path in file_paths:
            content = get_file_content(repo['full_name'], file_path)
            file_contents.append(content)
            logging.info(f'Added content from {file_path}')
    except Exception as e:
        logging.error(f'Error processing repository {repo["full_name"]}: {e}')

save_to_jsonl(file_contents)
logging.info('Finished processing all repositories')
