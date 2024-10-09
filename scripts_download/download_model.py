import os
import argparse
from huggingface_hub import HfApi, hf_hub_download

# 파일 다운로드 함수
def download_all_files_from_hf(api, repo_id, save_directory):
    """
    Hugging Face Hub에서 모든 파일을 다운로드하는 함수

    Args:
        repo_id (str): 레포지토리 ID
        save_directory (str): 파일을 저장할 디렉토리 경로
    """

    # 레포지토리 파일 목록 가져오기
    repo_files = api.list_repo_files(repo_id)

    for file_path in repo_files:
        # 파일 저장 경로 생성
        save_path = os.path.join("../", save_directory, file_path)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 파일 다운로드
        file_url = hf_hub_download(repo_id, file_path)
        with open(save_path, 'wb') as f:
            f.write(open(file_url, 'rb').read())

        print(f"Downloaded {file_path} to {save_path}")


def parsing():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="checkpoints_42dot_LLM-PLM-1.3B", help="Output directory")
    parser.add_argument("--repo-id", type=str, default="42dot/42dot_LLM-PLM-1.3B", help="Repository ID")
    args = parser.parse_args()

    return args


def main(args):
    """
    모델 체크포인트를 다운로드하는 메인 함수
    """

    # Hugging Face API 객체 생성
    api = HfApi()

    # 파일 다운로드 실행
    download_all_files_from_hf(api, args.repo_id, args.output_dir)


if __name__ == "__main__":
    args = parsing()
    main(args)
