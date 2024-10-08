from huggingface_hub import HfApi, hf_hub_download
import os

# Hugging Face API 객체 생성
api = HfApi()

# 다운로드하고자 하는 레포지토리 정보
repo_id = "42dot/42dot_LLM-PLM-1.3B"

# 저장할 디렉토리 설정
save_directory = "checkpoints_42dot_LLM-PLM-1.3B"


# 파일 다운로드 함수
def download_all_files_from_hf(repo_id, save_directory):
    # 레포지토리 파일 목록 가져오기
    repo_files = api.list_repo_files(repo_id)

    for file_path in repo_files:
        # 파일 저장 경로 생성
        save_path = os.path.join(save_directory, file_path)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 파일 다운로드
        file_url = hf_hub_download(repo_id, file_path)
        with open(save_path, 'wb') as f:
            f.write(open(file_url, 'rb').read())

        print(f"Downloaded {file_path} to {save_path}")

# 파일 다운로드 실행
download_all_files_from_hf(repo_id, save_directory)
